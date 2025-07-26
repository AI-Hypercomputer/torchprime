"""
Customized splash attention kernel wrapper. This is a varied copy
from the torch/xla repository. (https://github.com/pytorch/xla)
"""

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
  """
  Wrapper for calling Jax splash attention kernel with
  splashAttentionConfig. Currently only support forward pass.

  Args:
    mask: The attention mask. Can be a NumPy array, a JAX Array, or a
          `mask_lib.MultiHeadMask` object, depending on the masking
          strategy. Defines which elements can attend to others.
    query: The query tensor in PyTorch format. Shape typically
            `(batch_size, num_heads, q_seq_len, head_dim)`.
            It is made contiguous internally.
    key: The key tensor in PyTorch format. Shape typically
          `(batch_size, num_heads, kv_seq_len, head_dim)`.
          It is made contiguous internally.
    value: The value tensor in PyTorch format. Shape typically
            `(batch_size, num_heads, kv_seq_len, head_dim)`.
            It is made contiguous internally.
    config: A JSON string representing the `SplashAttentionConfig`. This
            configuration dictates parameters specific to the Splash Attention
            kernel (e.g., block sizes, kernel variants).
    decoder_segment_ids: Optional PyTorch tensor for segment IDs in decoder
                          attention, used for packed sequences. If provided,
                          it helps in preventing attention across segment boundaries.
    causal: A boolean indicating whether causal (autoregressive) masking
            should be applied. If True, a token can only attend to previous tokens.
    attn_logits_soft_cap: An optional float value to soft-cap the attention
                          logits, which can help in stabilizing training by
                          clipping large attention scores.
    is_forward: A boolean flag. If True, the function executes the forward pass.
                If False, it indicates an attempt to perform a backward pass,
                which is currently not supported.
    q_seq_shards: An integer specifying the number of shards the query sequence
                  is split into. This is relevant for distributed computation
                  and load balancing across devices.
    grad_output: This argument is expected for a backward pass but is currently
                  unused as only the forward pass is supported. It should be
                  a PyTorch tensor representing the gradient of the output
                  from a previous layer.

  Returns:
    A tuple containing:
    - output (torch.Tensor): The computed attention output tensor, also in
                              PyTorch format. This is the result of the
                              forward pass.
    - None: A placeholder for gradients with respect to query, indicating
            that the backward pass is not implemented.
    - None: A placeholder for gradients with respect to key and value,
            indicating that the backward pass is not implemented.

  Raises:
      (Implicit): The current implementation does not explicitly raise
      an error if `is_forward` is `False`, but returns an incomplete tuple.
      In a production system, it would be advisable to raise a `NotImplementedError`
      or similar for unsupported backward passes.

  Notes:
      - The function relies on external `call_jax` and `splash_attention_jax_wrapper`
        to interface with the JAX backend and execute the actual attention kernel.
      - Input `query`, `key`, and `value` tensors are converted to contiguous
        memory layout internally for potential performance benefits with the
        underlying kernel.
      - `config` string is parsed into a `SplashAttentionConfig` object.
      - This wrapper is designed for use in environments where JAX and PyTorch
        interoperate, typically on accelerators like TPUs.

  """
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
  """
  Wrapper for splash attention kernel with more customized input.

  Args:
    mask: An attention mask. It can be a NumPy array, a JAX Array, or a
          `mask_lib.MultiHeadMask` object. This defines the base masking
          logic (e.g., padding, visibility constraints).
    query: The query tensor for the attention mechanism. Expected shape
            `[batch, num_heads, q_seq_len, head_dim]`. It is expected to
            already be a JAX Array, potentially sharded externally.
    key: The key tensor for the attention mechanism. Expected shape
          `[batch, num_heads, kv_seq_len, head_dim]`.
          Expected to be a JAX Array.
    value: The value tensor for the attention mechanism. Expected shape
            `[batch, num_heads, kv_seq_len, head_dim]`.
            Expected to be a JAX Array.
    decoder_segment_ids: Optional JAX Array for decoder segment IDs. If
                          provided, it helps prevent attention across
                          packed sequence boundaries. If its shape is empty
                          or it's `None`, it's treated as not present.
    causal: A boolean indicating whether to apply causal (autoregressive)
            masking. If `True`, a token can only attend to preceding tokens
            in the sequence.
    config: An instance of `SplashAttentionConfig` containing various
            configuration parameters for the attention kernel, including
            mesh definition, sharding specifications, block sizes, and
            attention type.
    attn_logits_soft_cap: An optional float value to soft-cap the attention
                          logits. This can help stabilize training by
                          limiting the magnitude of attention scores.
    q_seq_shards: An integer representing the number of shards the query
                  sequence is divided into. This value influences the
                  calculation of `block_q` and `block_q_dkv` to ensure
                  proper sharding.

  Returns:
    jax.Array: The computed attention output tensor, sharded according to
                the configuration's `out_specs` (typically
                `P(("data", "fsdp"), "tensor", "context", None)`).

  Raises:
    AssertionError: If `query`'s sequence length does not match
                    `decoder_segment_ids.q`'s sequence length when
                    `decoder_segment_ids` is provided, indicating invalid
                    sharding along the sequence dimension.
    ValueError: If `config.attentiontype_local_sliding` is True but
                `config.slide_window_size` is not set.
    AssertionError: If the batch dimension of the query is not perfectly
                    shardable across the combined 'data' and 'fsdp' mesh axes.

  Notes:
    - The function dynamically creates the `jax.sharding.Mesh` from the
      `config.mesh` string.
    - Block sizes for various attention stages (e.g., `block_q`, `block_kv`)
      are determined by global configuration values and clamped by actual
      sequence lengths and query sequence shards.
    - Attention masks (`CausalMask`, `FullMask`, `LocalMask`, `MultiHeadMask`)
      are constructed based on `causal`, `config.attentiontype_local_sliding`,
      and the input `mask`.
    - The actual Splash Attention kernel is created using `splash_attention_kernel.make_splash_mha`.
    - `jax.vmap` is used with `shard_map.shard_map` to apply the kernel
      efficiently across the batch and potentially other dimensions, leveraging
      the defined mesh and sharding specifications.
    - This wrapper is designed to manage the complexities of distributed
      attention computation on TPUs, including data partitioning and
      device mapping.
  """
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
    # when q is sharded, we need ensure q block size is sharded by q_seq_shards
    block_q=min(global_block_q, seq_len // q_seq_shards),
    block_kv=min(global_block_kv, key.shape[2]),
    block_kv_compute=min(global_block_kv_compute, key.shape[2]),
    block_q_dkv=min(global_block_q_dkv, seq_len // q_seq_shards),
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
  named_sharding = jax.sharding.NamedSharding(mesh, P("tensor", "context"))
  axis_names_splash_kernel = splash_kernel.manual_sharding_spec(named_sharding)

  @functools.partial(
    shard_map.shard_map,
    mesh=mesh,
    in_specs=(
      P(("data", "fsdp"), "tensor", "context", None),
      axis_names,
      axis_names,
      # segment id sharding
      segment_axis_names,
      axis_names_splash_kernel,
    ),
    out_specs=P(("data", "fsdp"), "tensor", "context", None),
    check_rep=False,
  )
  def wrap_attention_kernel(query, key, value, decoder_segment_ids, splash_kernel):
    return jax.vmap(splash_kernel)(query, key, value, segment_ids=decoder_segment_ids)

  devices_in_data_fsdp = mesh.shape["data"] * mesh.shape["fsdp"]
  assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
    "Batch dimension should be shardable among the devices in data and fsdp axis"
  )
  x = wrap_attention_kernel(
    query,
    key,
    value,
    decoder_segment_ids,
    splash_kernel,
  )
  return x
