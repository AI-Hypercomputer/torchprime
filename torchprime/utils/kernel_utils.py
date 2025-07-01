import functools

import jax
import numpy as np
import torch
import torch_xla.debug.profiler as xp
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import (
  splash_attention_kernel,
  splash_attention_mask,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
  splash_attention_mask as mask_lib,
)
from jax.sharding import PartitionSpec as P
from torch_xla.core.xla_builder import call_jax
from torch_xla.distributed.spmd import Mesh
from torch_xla.experimental.splash_attention import (
  SplashAttentionConfig,
)


@xp.trace_me("tpu_splash_attention_jax_call_wrapper")
def tpu_splash_attention_jax_call_wrapper(
  mask: np.ndarray | jax.Array | mask_lib.MultiHeadMask,
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  config: str,
  decoder_segment_ids: torch.Tensor | None,
  causal: bool,
  attn_logits_soft_cap: float | None = None,
  is_forward: bool = True,
  q_seq_shards: int = 1,
  grad_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
  # return tuple to fit for the output num for both fwd and bwd
  query = query.contiguous()
  key = key.contiguous()
  value = value.contiguous()
  config = SplashAttentionConfig.from_json(config)
  input_args = [
    mask,
    query,
    key,
    value,
    decoder_segment_ids,
    causal,
    config,
    attn_logits_soft_cap,
    q_seq_shards,
  ]
  if is_forward:
    output = call_jax(
      splash_attention_jax_wrapper, input_args, {}, "splash_attention_jax_wrapper_fw"
    )
    return (output, None, None)
  else:
    # Only suppor forward pass for now
    return


@xp.trace_me("splash_attention_kernel_wrapper")
def splash_attention_jax_wrapper(
  mask: np.ndarray | jax.Array | mask_lib.MultiHeadMask,
  query,
  key,
  value,
  decoder_segment_ids,
  causal: bool,
  config: SplashAttentionConfig,
  attn_logits_soft_cap,
  q_seq_shards,
):
  mesh = Mesh.from_str(config.mesh).get_jax_mesh()
  # input q,k,v shape: [batch, #head, seq_len, head_dim]
  if decoder_segment_ids is not None and not decoder_segment_ids.shape:
    decoder_segment_ids = None
  if decoder_segment_ids is not None:
    decoder_segment_ids = splash_attention_kernel.SegmentIds(
      decoder_segment_ids, decoder_segment_ids
    )
  axis_names = jax.sharding.PartitionSpec(*config.qkv_partition_spec)
  segment_axis_names = jax.sharding.PartitionSpec(*config.segment_ids_partition_spec)

  global_block_q = config.sa_block_q
  global_block_kv = config.sa_block_kv
  global_block_kv_compute = config.sa_block_kv_compute
  global_block_q_dkv = config.sa_block_q_dkv
  global_block_kv_dkv = config.sa_block_kv_dkv
  global_block_kv_dkv_compute = config.sa_block_kv_dkv_compute
  global_block_q_dq = config.sa_block_q_dq
  global_block_kv_dq = config.sa_block_kv_dq
  global_use_fused_bwd_kernel = config.sa_use_fused_bwd_kernel
  global_q_layout = config.sa_q_layout
  global_k_layout = config.sa_k_layout
  global_v_layout = config.sa_v_layout

  seq_len = query.shape[2]
  if decoder_segment_ids is not None:
    assert seq_len == decoder_segment_ids.q.shape[1], (
      "Sharding along sequence dimension not allowed in tpu kernel attention"
    )
  block_sizes = splash_attention_kernel.BlockSizes(
    block_q=min(global_block_q, seq_len),
    block_kv=min(global_block_kv, key.shape[2]),
    block_kv_compute=min(global_block_kv_compute, key.shape[2]),
    block_q_dkv=min(global_block_q_dkv, seq_len),
    block_kv_dkv=min(global_block_kv_dkv, key.shape[2]),
    block_kv_dkv_compute=min(global_block_kv_dkv_compute, seq_len),
    block_q_dq=None if global_use_fused_bwd_kernel else min(global_block_q_dq, seq_len),
    block_kv_dq=None
    if global_use_fused_bwd_kernel
    else min(global_block_kv_dq, seq_len),
    use_fused_bwd_kernel=global_use_fused_bwd_kernel,
    q_layout=splash_attention_kernel.QKVLayout[global_q_layout],
    k_layout=splash_attention_kernel.QKVLayout[global_k_layout],
    v_layout=splash_attention_kernel.QKVLayout[global_v_layout],
  )
  if mask is None:
    if causal:
      mask = splash_attention_mask.CausalMask(shape=(seq_len, seq_len))
    else:
      mask = splash_attention_mask.FullMask(_shape=(seq_len, seq_len))

  # Apply local masking if local sliding attention is enabled.
  if config.attentiontype_local_sliding:
    if config.slide_window_size is None:
      raise ValueError(
        "Sliding_window_size must be set if Local Sliding attention type"
      )
    mask &= splash_attention_mask.LocalMask(
      shape=(seq_len, seq_len),
      window_size=(config.slide_window_size, config.slide_window_size),
      offset=0,
    )

  # Create multi-head mask
  multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])

  @functools.partial(
    jax.jit,
    static_argnames=[
      "multi_head_mask",
    ],
  )
  def wrap_splash_kernel(multi_head_mask):
    splash_kernel = splash_attention_kernel.make_splash_mha(
      mask=multi_head_mask,
      head_shards=1,
      q_seq_shards=q_seq_shards,
      block_sizes=block_sizes,
      attn_logits_soft_cap=attn_logits_soft_cap,
    )
    return splash_kernel

  # could add support for head sharding when needed
  splash_kernel = wrap_splash_kernel(multi_head_mask)
  kernel_sharding = P("context")
  # axis_names_splash_kernel = splash_kernel.manual_sharding_spec(kernel_sharding)

  @functools.partial(
    shard_map.shard_map,
    mesh=mesh,
    in_specs=(
      P(("data", "fsdp"), "context", None),
      axis_names,
      axis_names,
      # add support for segment id later
      segment_axis_names,
      kernel_sharding,
    ),
    out_specs=axis_names,
    check_rep=False,
  )
  def wrap_flash_attention(query, key, value, decoder_segment_ids):
    return jax.vmap(splash_kernel)(query, key, value, segment_ids=decoder_segment_ids)

  devices_in_data_fsdp = mesh.shape["data"] * mesh.shape["fsdp"]
  assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
    "Batch dimension should be shardable among the devices in data and fsdp axis"
  )
  x = wrap_flash_attention(
    mask=mask,
    query=query,
    key=key,
    value=value,
    decoder_segment_ids=decoder_segment_ids,
    q_seq_shards=q_seq_shards,
  )
  return x
