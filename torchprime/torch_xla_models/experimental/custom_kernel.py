import dataclasses
import functools
import json
import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any

import torch
import torch_xla
import torch_xla.core.xla_builder as xb
from torch.library import custom_op
from torch_xla.distributed.spmd import Mesh


@contextmanager
def _jax_env_context():
  try:
    os.environ['SKIP_MEGASCALE_PJRT_CLIENT'] = 'true'
    yield
  finally:
    os.environ.pop('SKIP_MEGASCALE_PJRT_CLIENT', None)


def requires_jax(func: Callable) -> Callable:
  """Decorator that ensures JAX is safely imported before function execution"""

  @functools.wraps(func)
  def wrapper(*args, **kwargs) -> Any:
    try:
      jax_import_guard()
    except ImportError as e:
      raise ImportError(
          "JAX import guard fail due to PJRT client is unavailable.") from e
    with _jax_env_context():
      return func(*args, **kwargs)

  return wrapper

def jax_import_guard():
  # Somehow, we need to grab the TPU before JAX locks it. Otherwise, any pt-xla TPU operations will hang.
  torch_xla._XLAC._init_computation_client()


@dataclasses.dataclass
class SplashAttentionConfig:
  sa_block_q: int
  sa_block_kv: int
  sa_block_kv_compute: int
  sa_block_q_dkv: int
  sa_block_kv_dkv: int
  sa_block_kv_dkv_compute: int
  sa_block_q_dq: int
  sa_block_kv_dq: int
  sa_use_fused_bwd_kernel: bool
  sa_q_layout: str
  sa_k_layout: str
  sa_v_layout: str
  mesh: str | None = None
  BATCH: str = "data"
  HEAD: str = "fsdp"
  LENGTH: str = "activation_length"
  D_KV: str = "activation_kv"
  flash_axis_names: tuple[str] = (BATCH, HEAD, LENGTH, D_KV)
  AttentionType_LOCAL_SLIDING: bool = False
  SLIDE_WINDOW_SIZE: int | None = None

  def to_json(self) -> str:
    """Serialize to JSON string"""
    return json.dumps(asdict(self))

  @classmethod
  def from_json(cls, data: str) -> "SplashAttentionConfig":
    """Deserialize from JSON string"""
    return SplashAttentionConfig(**json.loads(data))

  @requires_jax
  def get_jax_mesh(self):
    # TODO(zpcore, yifeit): Update the PyTorch/XLA xla_sharding.py class Mesh to
    # support mesh with the identical device_ids sequence. Below is a temporary
    # workaround.
    torch_xla_mesh = Mesh.from_str(self.mesh)
    # mesh.shape() will be in form of OrderedDict([('x', 4), ('y', 2)])
    mesh_shape = torch_xla_mesh.shape()
    keys = [k for k, _ in mesh_shape.items()]
    vals = [v for _, v in mesh_shape.items()]
    import jax
    return jax.make_mesh(vals, keys)

def splash_attention_jax_fun_wrapper(query, key, value, decoder_segment_ids, config: SplashAttentionConfig, attn_logits_soft_cap):
  """Splash attention kernel wrapper for JAX
  Inside the function, everything is JAX specific. We basically copy the
  function from
  https://github.com/AI-Hypercomputer/maxtext/blob/d8ffb5c4fc65e6832976226a8053236c2fe3164e/MaxText/layers/attentions.py#L336-L430
  """
  import jax
  from flax import linen as nn
  from jax.experimental import shard_map
  from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel,
    splash_attention_mask,
  )
  # q shape: [batch, #head, seq_len, kv]

  mesh = config.get_jax_mesh()

  if decoder_segment_ids is not None and not decoder_segment_ids.shape:
    decoder_segment_ids = None
  if decoder_segment_ids is not None:
    decoder_segment_ids = splash_attention_kernel.SegmentIds(decoder_segment_ids, decoder_segment_ids)
  axis_names = nn.logical_to_mesh_axes(config.flash_axis_names)
  segment_axis_names = nn.logical_to_mesh_axes((config.BATCH, "activation_length_no_heads"))

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
  shard_map = shard_map.shard_map
  @functools.partial(
      shard_map,
      mesh=mesh,
      in_specs=(
          axis_names,
          axis_names,
          axis_names,
          segment_axis_names,
      ),
      out_specs=axis_names,
      check_rep=False,
  )
  def wrap_flash_attention(query, key, value, decoder_segment_ids):
    if decoder_segment_ids is not None:
      assert (
          query.shape[2] == decoder_segment_ids.q.shape[1]
      ), "Sharding along sequence dimension not allowed in tpu kernel attention"
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=min(global_block_q, query.shape[2]),
        block_kv=min(global_block_kv, key.shape[2]),
        block_kv_compute=min(global_block_kv_compute, key.shape[2]),
        block_q_dkv=min(global_block_q_dkv, query.shape[2]),
        block_kv_dkv=min(global_block_kv_dkv, key.shape[2]),
        block_kv_dkv_compute=min(global_block_kv_dkv_compute, query.shape[2]),
        block_q_dq=None if global_use_fused_bwd_kernel else min(global_block_q_dq, query.shape[2]),
        block_kv_dq=None if global_use_fused_bwd_kernel else min(global_block_kv_dq, query.shape[2]),
        use_fused_bwd_kernel=global_use_fused_bwd_kernel,
        q_layout=splash_attention_kernel.QKVLayout[global_q_layout],
        k_layout=splash_attention_kernel.QKVLayout[global_k_layout],
        v_layout=splash_attention_kernel.QKVLayout[global_v_layout],
    )

    mask = splash_attention_mask.CausalMask(shape=(query.shape[2], query.shape[2]))

    # Apply local masking if local sliding attention is enabled.
    if config.AttentionType_LOCAL_SLIDING:
      if config.SLIDE_WINDOW_SIZE is None:
        raise ValueError("Sliding_window_size must be set if Local Sliding attention type")
      mask &= splash_attention_mask.LocalMask(
          shape=(query.shape[2], query.shape[2]),
          window_size=(config.SLIDE_WINDOW_SIZE, config.SLIDE_WINDOW_SIZE),
          offset=0,
      )

    # Create multi-head mask
    multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])
    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        head_shards=1,
        q_seq_shards=1,
        block_sizes=block_sizes,
        attn_logits_soft_cap=attn_logits_soft_cap,
    )
    return jax.vmap(splash_kernel)(query, key, value, segment_ids=decoder_segment_ids)

  devices_in_data_fsdp = mesh.shape["data"] * mesh.shape["fsdp"]
  assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
      "Batch dimension should be shardable among the devices in data and fsdp" " axis"
  )
  x = wrap_flash_attention(query, key, value, decoder_segment_ids)
  # x.shape = [batch, heads, seq_length, head_dim]
  return x

def tpu_splash_attention_jax_call_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    config: SplashAttentionConfig,
    decoder_segment_ids: torch.Tensor | None,
    attn_logits_soft_cap: float | None = None,
    is_forward: bool = True,
    grad_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
  # return tuple to fit for the output num for both fwd and bwd
  query = query.contiguous()
  key = key.contiguous()
  value = value.contiguous()
  import jax
  # TODO: xb.call_jax() doesn't accept the input tensor with shape size 0. We
  # have to split the decoder_segment_ids to be None or torch.Tensor cases.
  # Unify those two cases once 0 size shape tensor is supported.
  if decoder_segment_ids is not None and decoder_segment_ids.shape:
    jax_f = functools.partial(
      splash_attention_jax_fun_wrapper,
      config=config,
      attn_logits_soft_cap=attn_logits_soft_cap) 
    def jax_grad_f_wrapper(query, key, value, decoder_segment_ids, grad_output):
      primals, f_vjp = jax.vjp(jax_f, query, key, value, decoder_segment_ids)
      return f_vjp(grad_output)
    input_args = [query, key, value, decoder_segment_ids]
  else:
    jax_f = functools.partial(
      splash_attention_jax_fun_wrapper,
      decoder_segment_ids=None,
      config=config,
      attn_logits_soft_cap=attn_logits_soft_cap)
    def jax_grad_f_wrapper(query, key, value, grad_output):
      primals, f_vjp = jax.vjp(jax_f, query, key, value)
      return f_vjp(grad_output)
    input_args = [query, key, value]
  if is_forward:
    output = xb.call_jax(jax_f, input_args, {}, 'splash_attention_jax_fun_wrapper_fw')
    return (output, None, None)
  else:
    #TODO: find out a way to skip grad computation for decoder_segment_ids
    q_grad, k_grad, v_grad, *_rest = xb.call_jax(jax_grad_f_wrapper, input_args + [grad_output], {}, 'splash_attention_jax_fun_wrapper_bw')
    return (q_grad, k_grad, v_grad) 


@custom_op("xla::sa_custom_forward", mutates_args=())
def sa_custom_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, config: str, decoder_segment_ids: torch.Tensor | None, attn_logits_soft_cap: float | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  config = SplashAttentionConfig.from_json(config)
  return tpu_splash_attention_jax_call_wrapper(q, k, v, config, decoder_segment_ids, attn_logits_soft_cap, is_forward=True, grad_output=None)

@sa_custom_forward.register_fake
def sa_custom_forward_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, config: str, decoder_segment_ids: torch.Tensor | None, attn_logits_soft_cap: float | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  # q.shape: batch_size, seq_length, num_heads, kv (head_dim?)
  return (torch.empty_like(q), None, None)

@custom_op("xla::sa_custom_backward", mutates_args=())
def sa_custom_backward(grad_output: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, config: str, decoder_segment_ids: torch.Tensor | None, attn_logits_soft_cap: float | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  config = SplashAttentionConfig.from_json(config)
  o = tpu_splash_attention_jax_call_wrapper(q, k, v, config, decoder_segment_ids, attn_logits_soft_cap, is_forward=False, grad_output=grad_output)
  return o

@sa_custom_backward.register_fake
def sa_custom_backward_fake(grad_output: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, config: str, decoder_segment_ids: torch.Tensor | None, attn_logits_soft_cap: float | None)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  return (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))


class SplashAttention(torch.autograd.Function):

  @staticmethod
  @requires_jax
  def forward(ctx, q, k, v, config, decoder_segment_ids, attn_logits_soft_cap):
    output = sa_custom_forward(q, k, v, config, decoder_segment_ids, attn_logits_soft_cap)[0]
    ctx.save_for_backward(q, k, v, decoder_segment_ids, attn_logits_soft_cap)
    ctx.config = config
    return output

  @staticmethod
  @requires_jax
  def backward(ctx, grad_output):
    q, k, v, decoder_segment_ids, attn_logits_soft_cap  = ctx.saved_tensors
    config = ctx.config
    grad_q, grad_k, grad_v = sa_custom_backward(grad_output, q, k, v, config, decoder_segment_ids, attn_logits_soft_cap)
    return grad_q, grad_k, grad_v, None, None, None

def splash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    config: str,
    decoder_segment_ids: torch.Tensor | None = None,
    attn_logits_soft_cap: float | None = None,
) -> torch.Tensor:
  return SplashAttention.apply(q, k, v, config, decoder_segment_ids, attn_logits_soft_cap)
