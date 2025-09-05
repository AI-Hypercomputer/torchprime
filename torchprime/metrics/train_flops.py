"""Standalone FLOPs calculators
Mirroring ``maxtext_utils`` training formulas from MaxText repo:
https://github.com/AI-Hypercomputer/maxtext/blob/a8f6938c1a9048efa2dcd05c65b861a1ed96181b/MaxText/maxtext_utils.py#L478

The functions here reproduce the per-device training TFLOPs calculations from
``MaxText/maxtext_utils.py`` for a subset of decoder-only LLMs.  Only the core
pretraining FLOPs are included (no DPO or multimodal extras).
"""

from __future__ import annotations

from dataclasses import dataclass


def _ffn_flops(cfg, mlp_dim: int) -> int:
  """Matmul FLOPs for a dense FFN layer.
  https://github.com/AI-Hypercomputer/maxtext/blob/a8f6938c1a9048efa2dcd05c65b861a1ed96181b/MaxText/maxtext_utils.py#L300
  """
  ffn1 = (
    2
    * cfg.per_device_batch_size
    * cfg.max_target_length
    * mlp_dim
    * cfg.emb_dim
    * len(cfg.mlp_activations)
  )
  ffn2 = 2 * cfg.per_device_batch_size * cfg.max_target_length * mlp_dim * cfg.emb_dim
  return ffn1 + ffn2


def _get_dense_moe_layers(cfg) -> tuple[int, int]:
  """Get number of dense and MoE layers in interleaved architecture.
  https://github.com/AI-Hypercomputer/maxtext/blob/a8f6938c1a9048efa2dcd05c65b861a1ed96181b/MaxText/maxtext_utils.py#L328
  """
  if hasattr(cfg, "first_num_dense_layers"):
    num_dense = cfg.first_num_dense_layers
    num_moe = cfg.num_decoder_layers - cfg.first_num_dense_layers
  else:
    num_moe = cfg.num_decoder_layers // cfg.interleave_moe_layer_step
    num_dense = cfg.num_decoder_layers - num_moe
  return num_dense, num_moe


def _routed_and_shared_ffn_flops(cfg) -> int:
  """Matmul FLOPs for a mixture-of-experts FFN layer.
  https://github.com/AI-Hypercomputer/maxtext/blob/a8f6938c1a9048efa2dcd05c65b861a1ed96181b/MaxText/maxtext_utils.py#L315
  """
  gate = (
    2
    * cfg.per_device_batch_size
    * cfg.max_target_length
    * cfg.emb_dim
    * cfg.num_experts
  )
  num_dense, num_moe = _get_dense_moe_layers(cfg)
  dense = _ffn_flops(cfg, cfg.mlp_dim) * num_dense
  shared = _ffn_flops(cfg, cfg.moe_mlp_dim) * cfg.shared_experts
  routed = _ffn_flops(cfg, cfg.moe_mlp_dim) * cfg.num_experts_per_tok
  moe = (gate + shared + routed) * num_moe
  return dense + moe


def _mla_flops(cfg):
  """Matmul FLOPs for Multi-Head Latent Attention (i.e., Deepseek).
  https://github.com/AI-Hypercomputer/maxtext/blob/a8f6938c1a9048efa2dcd05c65b861a1ed96181b/MaxText/maxtext_utils.py#L268
  """
  batch_len = cfg.per_device_batch_size * cfg.max_target_length
  qk_sum = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
  if cfg.q_lora_rank == 0:
    q_flops = 2 * batch_len * cfg.emb_dim * cfg.num_query_heads * qk_sum
  else:
    q_flops = (
      2
      * batch_len
      * (cfg.emb_dim * cfg.q_lora_rank + cfg.q_lora_rank * cfg.num_query_heads * qk_sum)
    )
  kv_flops = (
    2
    * batch_len
    * (
      cfg.emb_dim * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)
      + cfg.kv_lora_rank * cfg.num_query_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim)
    )
  )
  qkv_flops = q_flops + kv_flops
  attn_flops = (
    2
    * batch_len
    * cfg.max_target_length
    * cfg.num_query_heads
    * (qk_sum + cfg.v_head_dim)
  )
  proj_flops = 2 * batch_len * cfg.emb_dim * cfg.num_query_heads * cfg.v_head_dim
  return qkv_flops, attn_flops, proj_flops


def _chunked_attention_flops_per_layer(cfg, seq_len: int, chunk: int) -> int:
  """non-causal FLOPs for a single layer of chunked attention (in Llama4)
  https://github.com/AI-Hypercomputer/maxtext/blob/a8f6938c1a9048efa2dcd05c65b861a1ed96181b/MaxText/maxtext_utils.py#L223
  """
  num_chunks = seq_len // chunk
  rem = seq_len % chunk
  complexity = num_chunks * chunk**2 + rem**2
  return 4 * cfg.per_device_batch_size * complexity * cfg.num_query_heads * cfg.head_dim


def _llama4_attention_tflops(cfg) -> float:
  """Llama4's specific attention-only TFLOPs, i.e., alternating global and chunked attention layers
  https://github.com/AI-Hypercomputer/maxtext/blob/a8f6938c1a9048efa2dcd05c65b861a1ed96181b/MaxText/maxtext_utils.py#L234
  """
  num_layers = cfg.num_decoder_layers
  seq_len = cfg.max_target_length
  chunk = cfg.chunk_attn_window_size
  num_global = num_layers // cfg.nope_layer_interval
  num_chunked = num_layers - num_global
  global_flops = (
    4 * cfg.per_device_batch_size * seq_len**2 * cfg.num_query_heads * cfg.head_dim
  )
  chunked_flops = _chunked_attention_flops_per_layer(cfg, seq_len, chunk)
  noncausal = num_global * global_flops + num_chunked * chunked_flops
  causal = noncausal / 2
  return causal * 3 / 10**12


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LlamaConfig:
  per_device_batch_size: int
  max_target_length: int
  num_decoder_layers: int
  emb_dim: int
  num_query_heads: int
  num_kv_heads: int
  head_dim: int
  mlp_dim: int
  vocab_size: int
  mlp_activations: tuple[str, ...]
  gradient_accumulation_steps: int


@dataclass
class Llama4Config:
  per_device_batch_size: int
  max_target_length: int
  num_decoder_layers: int
  emb_dim: int
  num_query_heads: int
  num_kv_heads: int
  head_dim: int
  mlp_dim: int
  vocab_size: int
  mlp_activations: tuple[str, ...]
  gradient_accumulation_steps: int
  moe_mlp_dim: int
  num_experts: int
  num_experts_per_tok: int
  shared_experts: int
  interleave_moe_layer_step: int
  chunk_attn_window_size: int
  nope_layer_interval: int


@dataclass
class DeepSeekConfig:
  per_device_batch_size: int
  max_target_length: int
  num_decoder_layers: int
  emb_dim: int
  num_query_heads: int
  num_kv_heads: int
  head_dim: int
  mlp_dim: int
  vocab_size: int
  mlp_activations: tuple[str, ...]
  gradient_accumulation_steps: int
  moe_mlp_dim: int
  num_experts: int
  num_experts_per_tok: int
  shared_experts: int
  first_num_dense_layers: int
  qk_nope_head_dim: int
  qk_rope_head_dim: int
  v_head_dim: int
  q_lora_rank: int
  kv_lora_rank: int


# ---------------------------------------------------------------------------


def llama3_style_models_tflops(cfg: LlamaConfig):
  """Training TFLOPs per device for Llama3-style models.
  https://github.com/AI-Hypercomputer/maxtext/blob/a8f6938c1a9048efa2dcd05c65b861a1ed96181b/MaxText/maxtext_utils.py#L478
  """
  total_ffn = _ffn_flops(cfg, cfg.mlp_dim)
  qkv = (
    2
    * cfg.per_device_batch_size
    * cfg.max_target_length
    * cfg.emb_dim
    * (cfg.num_query_heads + 2 * cfg.num_kv_heads)
    * cfg.head_dim
  )
  noncausal_attn = (
    4
    * cfg.per_device_batch_size
    * cfg.max_target_length**2
    * cfg.num_query_heads
    * cfg.head_dim
  )
  proj = (
    2
    * cfg.per_device_batch_size
    * cfg.max_target_length
    * cfg.emb_dim
    * cfg.num_query_heads
    * cfg.head_dim
  )
  causal_attn = noncausal_attn / 2  # Divide attention flops by 2 due to causal mask
  embed = (
    2 * cfg.per_device_batch_size * cfg.max_target_length * cfg.emb_dim * cfg.vocab_size
  )
  lw_tflops = ((total_ffn + qkv + proj) * cfg.num_decoder_layers + embed) * 3 / 10**12
  attn_tflops = causal_attn * cfg.num_decoder_layers * 3 / 10**12
  lw_tflops *= cfg.gradient_accumulation_steps
  attn_tflops *= cfg.gradient_accumulation_steps
  return (
    lw_tflops + attn_tflops,
    lw_tflops,
    attn_tflops,
  )


def llama4_tflops(cfg: Llama4Config):
  """Training TFLOPs per device for Llama4 models.
  https://github.com/AI-Hypercomputer/maxtext/blob/a8f6938c1a9048efa2dcd05c65b861a1ed96181b/MaxText/maxtext_utils.py#L478
  """
  if cfg.num_experts > 1:
    total_ffn = _routed_and_shared_ffn_flops(cfg)
  else:
    total_ffn = _ffn_flops(cfg, cfg.mlp_dim)
  qkv = (
    2
    * cfg.per_device_batch_size
    * cfg.max_target_length
    * cfg.emb_dim
    * (cfg.num_query_heads + 2 * cfg.num_kv_heads)
    * cfg.head_dim
  )
  proj = (
    2
    * cfg.per_device_batch_size
    * cfg.max_target_length
    * cfg.emb_dim
    * cfg.num_query_heads
    * cfg.head_dim
  )
  embed = (
    2 * cfg.per_device_batch_size * cfg.max_target_length * cfg.emb_dim * cfg.vocab_size
  )
  attn_tflops = _llama4_attention_tflops(cfg)
  lw_tflops = (total_ffn + (qkv + proj) * cfg.num_decoder_layers + embed) * 3 / 10**12
  lw_tflops *= cfg.gradient_accumulation_steps
  attn_tflops *= cfg.gradient_accumulation_steps
  return lw_tflops + attn_tflops, lw_tflops, attn_tflops


def deepseek_tflops(cfg: DeepSeekConfig):
  """Training TFLOPs per device for Deepseek models.
  https://github.com/AI-Hypercomputer/maxtext/blob/a8f6938c1a9048efa2dcd05c65b861a1ed96181b/MaxText/maxtext_utils.py#L478
  """
  if cfg.num_experts > 1:
    total_ffn = _routed_and_shared_ffn_flops(cfg)
  else:
    total_ffn = _ffn_flops(cfg, cfg.mlp_dim)
  qkv, noncausal_attn, proj = _mla_flops(cfg)
  causal_attn = noncausal_attn / 2
  embed = (
    2 * cfg.per_device_batch_size * cfg.max_target_length * cfg.emb_dim * cfg.vocab_size
  )
  lw_tflops = (total_ffn + (qkv + proj) * cfg.num_decoder_layers + embed) * 3 / 10**12
  attn_tflops = causal_attn * cfg.num_decoder_layers * 3 / 10**12
  lw_tflops *= cfg.gradient_accumulation_steps
  attn_tflops *= cfg.gradient_accumulation_steps
  return lw_tflops + attn_tflops, lw_tflops, attn_tflops
