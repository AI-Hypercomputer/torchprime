"""PyTorch DeepSeek V3 model for supervised fine-tuning.

This implementation mirrors the HuggingFace architecture but only includes
features required for the unit tests. It reuses the building blocks from
``torchprime`` used for the Llama models.
"""

from __future__ import annotations

import math

import torch
import torch_xla.debug.profiler as xp
from omegaconf import DictConfig
from torch import nn
from transformers.activations import ACT2FN

from torchprime.torch_xla_models import offloading
from torchprime.torch_xla_models.attention import AttentionModule, repeat_kv
from torchprime.torch_xla_models.loss import cross_entropy_loss
from torchprime.torch_xla_models.model.base_causal_lm import BaseCausalLM


class DeepseekV3RMSNorm(nn.Module):
  """RMSNorm used throughout the model."""

  def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


class DeepseekV3RotaryEmbedding(nn.Module):
  """Rotary positional embedding used for queries and keys."""

  inv_freq: torch.Tensor

  def __init__(self, head_dim: int, rope_theta: float) -> None:
    super().__init__()
    inv_freq = 1.0 / (
      rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    self.register_buffer("inv_freq", inv_freq, persistent=False)

  @torch.no_grad()
  def forward(
    self, x: torch.Tensor, position_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq_expanded = self.inv_freq[None, :, None].expand(
      position_ids.shape[0], -1, 1
    )
    pos_emb = (inv_freq_expanded * position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((pos_emb, pos_emb), dim=-1)
    cos = emb.cos().to(dtype=x.dtype)
    sin = emb.sin().to(dtype=x.dtype)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
  q: torch.Tensor,
  k: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  position_ids: torch.Tensor | None = None,
  unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
  cos = cos.unsqueeze(unsqueeze_dim)
  sin = sin.unsqueeze(unsqueeze_dim)
  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed


class DeepseekV3MLP(nn.Module):
  def __init__(self, config: DictConfig) -> None:
    super().__init__()
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    self.act_fn = nn.SiLU() if config.hidden_act == "silu" else ACT2FN[config.hidden_act]

  @xp.trace_me("DeepseekV3MLP")
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj


class DeepseekV3Attention(nn.Module):
  """Minimal multi-head self-attention for DeepSeek V3."""

  def __init__(self, config: DictConfig, layer_idx: int | None = None) -> None:
    super().__init__()
    self.config = config
    self.attention_block = AttentionModule(config)
    self.layer_idx = layer_idx

    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.head_dim = config.hidden_size // self.num_heads
    self.rope_theta = config.rope_theta
    self.attention_dropout = config.attention_dropout
    self.is_causal = True

    self.q_proj = nn.Linear(
      self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
    )
    self.k_proj = nn.Linear(
      self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
    )
    self.v_proj = nn.Linear(
      self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
    )
    self.o_proj = nn.Linear(
      self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
    )

    head_dim = getattr(config, "head_dim", self.head_dim)
    self.rotary_emb = DeepseekV3RotaryEmbedding(head_dim, self.rope_theta)

  @xp.trace_me("DeepseekV3Attention")
  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
  ) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(
      batch_size, seq_len, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(
      batch_size, seq_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(
      batch_size, seq_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    if position_embeddings is None:
      cos, sin = self.rotary_emb(value_states, position_ids)
    else:
      cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
      query_states, key_states, cos, sin, position_ids
    )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    attn_weights = attn_weights / math.sqrt(self.head_dim)
    if attention_mask is not None:
      attn_weights = attn_weights + attention_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
    attn_output = self.o_proj(attn_output)
    return attn_output


class DeepseekV3DecoderLayer(nn.Module):
  def __init__(self, config: DictConfig, layer_idx: int) -> None:
    super().__init__()
    self.hidden_size = config.hidden_size
    self.self_attn = DeepseekV3Attention(config=config, layer_idx=layer_idx)
    self.mlp = DeepseekV3MLP(config)
    self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
      hidden_states=hidden_states,
      attention_mask=attention_mask,
      position_ids=position_ids,
      position_embeddings=position_embeddings,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


class DeepseekV3Model(nn.Module):
  """Transformer decoder composed of ``DeepseekV3DecoderLayer`` blocks."""

  def __init__(self, config: DictConfig) -> None:
    super().__init__()
    self.vocab_size = config.vocab_size
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    self.layers = nn.ModuleList(
      [DeepseekV3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
    )
    self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.rope_theta = config.rope_theta
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    self.rotary_emb = DeepseekV3RotaryEmbedding(head_dim, self.rope_theta)

  @xp.trace_me("DeepseekV3Model")
  def forward(
    self,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    inputs_embeds = self.embed_tokens(input_ids)
    seq_len = inputs_embeds.size(1)
    position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).float()

    causal_mask = torch.triu(
      torch.full((seq_len, seq_len), float("-inf"), device=inputs_embeds.device), 1
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    if attention_mask is not None:
      causal_mask = causal_mask * attention_mask[:, None, None, :]

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    for layer in self.layers:
      hidden_states = layer(
        hidden_states,
        attention_mask=causal_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
      )
    hidden_states = self.norm(hidden_states)
    return hidden_states


class DeepseekForCausalLM(BaseCausalLM):
  """DeepSeek V3 model wrapper for causal language modeling."""

  def __init__(self, config: DictConfig) -> None:
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
    logits = self.lm_head(hidden_states).float()
    if labels is None:
      return logits, None
    loss = cross_entropy_loss(logits, labels=labels, vocab_size=self.config.vocab_size)
    return logits, loss
