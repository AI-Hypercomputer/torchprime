"""PyTorch/XLA Deepseek v3 model.

Following the Deepseek v3 implementation from HF transformers
https://github.com/huggingface/transformers/blob/18a7c29ff8431193887e1065777e9cde29d46e53/src/transformers/models/deepseek_v3/modular_deepseek_v3.py
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torch_xla.debug.profiler as xp
from omegaconf import DictConfig
from torch import nn
from torch_xla.experimental.custom_kernel import GMM
from transformers.activations import ACT2FN
from transformers.utils import logging

from torchprime.layers.sequential import HomogeneousSequential
from torchprime.rope.rope import deepseek_v3_rope_init_fn
from torchprime.torch_xla_models import offloading
from torchprime.torch_xla_models.attention import AttentionModule
from torchprime.torch_xla_models.loss import cross_entropy_loss
from torchprime.torch_xla_models.model.base_causal_lm import BaseCausalLM
from torchprime.torch_xla_models.model.llama.model import apply_rotary_pos_emb

logger = logging.get_logger(__name__)
FP32 = torch.float32


class DeepseekV3RMSNorm(nn.Module):
  def __init__(self, hidden_size: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  @xp.trace_me("DeepseekV3RMSNorm")
  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(FP32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


class DeepseekV3RotaryEmbedding(nn.Module):
  inv_freq: nn.Buffer

  def __init__(self, config: DictConfig):
    super().__init__()
    self.config = config
    inv_freq, self.attention_scaling = deepseek_v3_rope_init_fn(self.config)
    self.register_buffer("inv_freq", inv_freq.to(FP32), persistent=False)
    self.original_inv_freq = self.inv_freq

  @torch.no_grad()
  def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
    inv_freq_expanded = (
      self.inv_freq[None, :, None].to(FP32).expand(position_ids.shape[0], -1, 1)
    )
    position_ids_expanded = position_ids[:, None, :].to(FP32)

    device_type = x.device.type
    device_type = (
      device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    )
    with torch.autocast(device_type=device_type, enabled=False):
      freqs = (inv_freq_expanded.to(FP32) @ position_ids_expanded.to(FP32)).transpose(
        1, 2
      )
      emb = torch.cat((freqs, freqs), dim=-1)
      cos = emb.cos() * self.attention_scaling
      sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_interleave(
  q: torch.Tensor,
  k: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  position_ids: torch.Tensor | None = None,
  unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)

  b, h, s, d = q.shape
  q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

  b, h, s, d = k.shape
  k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
  if scale <= 1:
    return 1.0
  return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV3MLP(nn.Module):
  def __init__(
    self,
    config: DictConfig,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
  ):
    super().__init__()
    self.config = config
    self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
    self.intermediate_size = (
      config.intermediate_size if intermediate_size is None else intermediate_size
    )

    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    self.act_fn = ACT2FN[config.hidden_act]

  @xp.trace_me("DeepseekV3MLP")
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj


class DeepseekV3TopkRouter(nn.Module):
  def __init__(self, config: DictConfig):
    super().__init__()
    self.config = config
    self.top_k = config.num_experts_per_tok
    self.n_routed_experts = config.n_routed_experts
    self.routed_scaling_factor = config.routed_scaling_factor
    self.n_group = config.n_group
    self.topk_group = config.topk_group
    self.norm_topk_prob = config.norm_topk_prob

    self.weight = nn.Parameter(
      torch.empty((self.n_routed_experts, config.hidden_size), dtype=FP32)
    )
    self.register_buffer(
      "e_score_correction_bias", torch.zeros(self.n_routed_experts, dtype=FP32)
    )

  @torch.no_grad()
  def get_topk_indices(self, scores: torch.Tensor) -> torch.Tensor:
    scores_for_choice = scores.view(
      -1, self.n_routed_experts
    ) + self.e_score_correction_bias.unsqueeze(0)
    group_scores = (
      scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
      .topk(2, dim=-1)[0]
      .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
      group_mask.unsqueeze(-1)
      .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
      .reshape(-1, self.n_routed_experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
    return topk_indices

  @xp.trace_me("DeepseekV3TopkRouter")
  def forward(self, hidden_states: torch.Tensor):
    hidden_states = hidden_states.view(-1, self.config.hidden_size)
    router_logits = F.linear(hidden_states.to(FP32), self.weight.to(FP32))
    scores = router_logits.sigmoid()
    topk_indices = self.get_topk_indices(scores)
    topk_weights = scores.gather(1, topk_indices)
    if self.norm_topk_prob:
      denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
      topk_weights /= denominator
    topk_weights = topk_weights * self.routed_scaling_factor
    return topk_indices, topk_weights


class GroupedMoEWeights(nn.Module):
  """Grouped expert weights that can be sharded along the expert dim (E)."""

  def __init__(self, E: int, D: int, H: int, dtype: torch.dtype):
    super().__init__()
    self.W_gate = nn.Parameter(torch.empty(E, D, H, dtype=dtype))
    self.W_up = nn.Parameter(torch.empty(E, D, H, dtype=dtype))
    self.W_down = nn.Parameter(torch.empty(E, H, D, dtype=dtype))
    nn.init.kaiming_uniform_(self.W_gate, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.W_up, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.W_down, a=math.sqrt(5))


class DeepseekV3MoE(nn.Module):
  """
  Mixture-of-Experts, with a conditional switch for the GMM kernel.
  This version does NOT perform token dropping.
  """

  def __init__(self, config: DictConfig):
    super().__init__()
    self.config = config
    self.E = config.n_routed_experts
    self.K = config.num_experts_per_tok
    self.D = config.hidden_size
    self.I = config.moe_intermediate_size

    # Add a flag to control kernel usage, defaulting to True for TPU performance
    self.use_gmm_kernel_for_moe = config.get("use_gmm_kernel_for_moe", True)

    self.gate = DeepseekV3TopkRouter(config)
    self.grouped = GroupedMoEWeights(self.E, self.D, self.I, dtype=torch.bfloat16)
    self.shared_experts = DeepseekV3MLP(
      config=config, intermediate_size=self.I * config.n_shared_experts
    )
    self.act_fn = ACT2FN[config.hidden_act]

  # The weight loading functions remain the same for checkpoint compatibility
  @torch.no_grad()
  def _pre_load_old_keys(self, state_dict, prefix: str):
    has_old = any(
      k.startswith(prefix + "experts.0.gate_proj.weight") for k in state_dict
    )
    if not has_old:
      return
    E = self.E
    Wg = torch.stack(
      [state_dict[f"{prefix}experts.{e}.gate_proj.weight"].t() for e in range(E)], dim=0
    )
    Wu = torch.stack(
      [state_dict[f"{prefix}experts.{e}.up_proj.weight"].t() for e in range(E)], dim=0
    )
    Wd = torch.stack(
      [state_dict[f"{prefix}experts.{e}.down_proj.weight"].t() for e in range(E)], dim=0
    )
    Wg = Wg.to(self.grouped.W_gate.dtype)
    Wu = Wu.to(self.grouped.W_up.dtype)
    Wd = Wd.to(self.grouped.W_down.dtype)
    self.grouped.W_gate.copy_(Wg.contiguous())
    self.grouped.W_up.copy_(Wu.contiguous())
    self.grouped.W_down.copy_(Wd.contiguous())

  @torch.no_grad()
  def _post_state_dict_old_keys(self, state_dict, prefix: str):
    E = self.E
    for e in range(E):
      state_dict[f"{prefix}experts.{e}.gate_proj.weight"] = (
        self.grouped.W_gate[e].t().contiguous().to(FP32)
      )
      state_dict[f"{prefix}experts.{e}.up_proj.weight"] = (
        self.grouped.W_up[e].t().contiguous().to(FP32)
      )
      state_dict[f"{prefix}experts.{e}.down_proj.weight"] = (
        self.grouped.W_down[e].t().contiguous().to(FP32)
      )

  def _grouped_weights(self, dtype: torch.dtype):
    return (
      self.grouped.W_gate.to(dtype),
      self.grouped.W_up.to(dtype),
      self.grouped.W_down.to(dtype),
    )

  def _cpu_gmm(
    self, x: torch.Tensor, W: torch.Tensor, group_sizes: torch.Tensor
  ) -> torch.Tensor:
    """CPU-friendly implementation of Grouped Matrix Multiply."""
    # Split the input tensor into a list of tensors, one for each expert group.
    # We filter out groups of size 0 to avoid errors with torch.split.
    non_zero_sizes = [size for size in group_sizes.tolist() if size > 0]
    expert_inputs = torch.split(x, non_zero_sizes, dim=0)

    output_chunks = []
    input_idx = 0
    for e in range(self.E):
      if group_sizes[e] > 0:
        # Perform standard matrix multiplication for the current expert
        output_chunks.append(torch.matmul(expert_inputs[input_idx], W[e]))
        input_idx += 1

    return torch.cat(output_chunks, dim=0)

  @xp.trace_me("DeepseekV3MoE")
  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    B, S, D = hidden_states.shape
    T = B * S
    x = hidden_states.view(T, self.D)

    topk_idx, topk_w = self.gate(x)  # [T, K], [T, K]
    topk_w = topk_w.to(x.dtype)
    expert_ids = topk_idx.view(-1)

    expert_one_hot = F.one_hot(expert_ids, num_classes=self.E)
    group_sizes = expert_one_hot.sum(dim=0).to(torch.int32)

    _, sort_ix = torch.sort(expert_ids)
    replicated_x = x.unsqueeze(1).expand(-1, self.K, -1).reshape(-1, self.D)
    sorted_x = replicated_x[sort_ix]

    W_gate, W_up, W_down = self._grouped_weights(x.dtype)

    # Conditional logic to switch between GMM kernel and CPU fallback
    if self.use_gmm_kernel_for_moe and GMM is not None:
      gate_out = GMM.apply(sorted_x, W_gate, group_sizes)
      up_out = GMM.apply(sorted_x, W_up, group_sizes)
      act_out = self.act_fn(gate_out) * up_out
      mlp_output_sorted = GMM.apply(act_out, W_down, group_sizes)
    else:
      if GMM is None and self.use_gmm_kernel_for_moe:
        logger.warning_once(
          "GMM custom kernel not available. Falling back to CPU-friendly MoE implementation. "
          "This will be slower. To disable this warning, set `use_gmm_kernel_for_moe: false` in your config."
        )

      gate_out = self._cpu_gmm(sorted_x, W_gate, group_sizes)
      up_out = self._cpu_gmm(sorted_x, W_up, group_sizes)
      act_out = self.act_fn(gate_out) * up_out
      mlp_output_sorted = self._cpu_gmm(act_out, W_down, group_sizes)

    mlp_output_unsorted = torch.empty_like(mlp_output_sorted)
    mlp_output_unsorted[sort_ix] = mlp_output_sorted

    weighted_output = (
      mlp_output_unsorted.view(T, self.K, self.D) * topk_w.unsqueeze(-1)
    ).sum(dim=1)

    out = weighted_output.view(B, S, D) + self.shared_experts(hidden_states)
    return out


class DeepseekV3Attention(nn.Module):
  """Multi-headed latent attention."""

  def __init__(self, config: DictConfig, layer_idx: int | None = None):
    super().__init__()
    self.config = config
    self.attention_block = AttentionModule(config)
    self.layer_idx = layer_idx
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.attention_dropout = (
      config.attention_dropout
    )  # this is not used in the current implementation
    self.num_heads = config.num_attention_heads
    self.rope_theta = config.rope_theta
    #############
    self.q_lora_rank = config.q_lora_rank
    self.qk_rope_head_dim = config.qk_rope_head_dim
    self.kv_lora_rank = config.kv_lora_rank
    self.v_head_dim = config.v_head_dim
    self.qk_nope_head_dim = config.qk_nope_head_dim
    #############
    self.qk_head_dim = config.qk_head_dim

    self.is_causal = True
    if config.q_lora_rank is None:
      self.q_proj = nn.Linear(
        config.hidden_size, self.num_heads * self.qk_head_dim, bias=False
      )
    else:
      self.q_a_proj = nn.Linear(
        config.hidden_size, config.q_lora_rank, bias=config.attention_bias
      )
      self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
      self.q_b_proj = nn.Linear(
        config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
      )

    self.kv_a_proj_with_mqa = nn.Linear(
      config.hidden_size,
      config.kv_lora_rank + config.qk_rope_head_dim,
      bias=config.attention_bias,
    )
    self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
    self.kv_b_proj = nn.Linear(
      config.kv_lora_rank,
      self.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
      bias=False,
    )

    self.o_proj = nn.Linear(
      self.num_heads * config.v_head_dim, config.hidden_size, bias=config.attention_bias
    )

    self.scaling = self.qk_head_dim ** (-0.5)
    if config.rope_scaling is not None:
      mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
      scaling_factor = config.rope_scaling["factor"]
      if mscale_all_dim:
        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        self.scaling = self.scaling * mscale * mscale

  @xp.trace_me("DeepseekV3Attention")
  def forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
  ) -> torch.Tensor:
    batch_size, seq_length = hidden_states.shape[:2]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
    key_shape = (
      batch_size,
      seq_length,
      -1,
      self.config.qk_nope_head_dim + self.config.v_head_dim,
    )

    if self.config.q_lora_rank is None:
      q_states = self.q_proj(hidden_states)
    else:
      q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q_states = q_states.view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(
      q_states, [self.config.qk_nope_head_dim, self.config.qk_rope_head_dim], dim=-1
    )

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(
      compressed_kv, [self.config.kv_lora_rank, self.config.qk_rope_head_dim], dim=-1
    )

    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
    k_pass, value_states = torch.split(
      k_pass, [self.config.qk_nope_head_dim, self.config.v_head_dim], dim=-1
    )

    k_rot = k_rot.view(batch_size, 1, seq_length, self.config.qk_rope_head_dim)
    cos, sin = position_embeddings
    if self.config.rope_interleave:
      q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    else:
      q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)

    attn_output = self.attention_block(
      query_states, key_states, value_states, attention_mask
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_length, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output


class DeepseekV3DecoderLayer(nn.Module):
  def __init__(self, config: DictConfig, layer_idx: int):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.self_attn = DeepseekV3Attention(config=config, layer_idx=layer_idx)
    if layer_idx >= config.first_k_dense_replace:
      self.mlp = DeepseekV3MoE(config)
    else:
      self.mlp = DeepseekV3MLP(config)
    self.input_layernorm = DeepseekV3RMSNorm(
      config.hidden_size, eps=config.rms_norm_eps
    )
    self.post_attention_layernorm = DeepseekV3RMSNorm(
      config.hidden_size, eps=config.rms_norm_eps
    )

  @xp.trace_me("DeepseekV3DecoderLayer")
  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
  ) -> torch.Tensor:
    hidden_states = offloading.offload_name(hidden_states, "decoder_input")
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(
      hidden_states, position_embeddings, attention_mask, position_ids
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


class DeepseekV3Model(nn.Module):
  def __init__(self, config: DictConfig):
    super().__init__()
    self.vocab_size = config.vocab_size
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    self.layers = HomogeneousSequential(
      *[
        DeepseekV3DecoderLayer(config, layer_idx)
        for layer_idx in range(config.num_hidden_layers)
      ]
    )
    self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.rotary_emb = DeepseekV3RotaryEmbedding(config=config)

  @xp.trace_me("DeepseekV3Model")
  def forward(
    self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None
  ) -> torch.Tensor:
    inputs_embeds = self.embed_tokens(input_ids)
    seq_length = inputs_embeds.size(1)
    position_ids = (
      torch.arange(seq_length, device=inputs_embeds.device).unsqueeze(0).to(FP32)
    )

    causal_mask = torch.triu(
      torch.full((seq_length, seq_length), float("-inf"), device=inputs_embeds.device),
      diagonal=1,
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    if attention_mask is not None:
      causal_mask = causal_mask * attention_mask[:, None, None, :]

    position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
    hidden_states = self.layers(
      inputs_embeds,
      attention_mask=causal_mask,
      position_ids=position_ids,
      position_embeddings=position_embeddings,
    )
    hidden_states = self.norm(hidden_states)
    return hidden_states


class DeepseekV3ForCausalLM(BaseCausalLM):
  def __init__(self, config: DictConfig):
    super().__init__()
    self.config = config
    self.model = DeepseekV3Model(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.apply(self._init_weights)

  @xp.trace_me("DeepseekV3ForCausalLM")
  def forward(
    self,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)
    logits = self.lm_head(hidden_states)
    logits = logits.to(FP32)
    if labels is None:
      return logits, None
    loss = cross_entropy_loss(logits, labels=labels, vocab_size=self.config.vocab_size)
    return logits, loss
