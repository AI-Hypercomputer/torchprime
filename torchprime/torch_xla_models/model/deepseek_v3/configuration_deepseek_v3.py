"""Configuration class for DeepSeek V3."""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class DeepseekV3Config(PretrainedConfig):
  """Configuration for DeepSeek V3 models.

  This mirrors the parameters from the Hugging Face implementation but only
  includes attributes relevant for supervised fine-tuning on TPU.
  """

  model_type = "deepseek_v3"
  keys_to_ignore_at_inference = ["past_key_values"]

  def __init__(
    self,
    vocab_size: int = 129280,
    hidden_size: int = 7168,
    intermediate_size: int = 18432,
    moe_intermediate_size: int = 2048,
    num_hidden_layers: int = 61,
    num_attention_heads: int = 128,
    num_key_value_heads: int | None = None,
    n_shared_experts: int = 1,
    n_routed_experts: int = 256,
    routed_scaling_factor: float = 2.5,
    kv_lora_rank: int = 512,
    q_lora_rank: int = 1536,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    qk_nope_head_dim: int = 128,
    n_group: int = 8,
    topk_group: int = 4,
    num_experts_per_tok: int | None = 8,
    first_k_dense_replace: int = 3,
    norm_topk_prob: bool = True,
    hidden_act: str = "silu",
    max_position_embeddings: int = 4096,
    initializer_range: float = 0.02,
    rms_norm_eps: float = 1e-6,
    use_cache: bool = True,
    pad_token_id: int | None = None,
    bos_token_id: int = 0,
    eos_token_id: int = 1,
    pretraining_tp: int = 1,
    tie_word_embeddings: bool = False,
    rope_theta: float = 10000.0,
    rope_scaling: dict | None = None,
    rope_interleave: bool = True,
    attention_bias: bool = False,
    attention_dropout: float = 0.0,
    **kwargs,
  ) -> None:
    if num_key_value_heads is None:
      num_key_value_heads = num_attention_heads

    self.vocab_size = vocab_size
    self.max_position_embeddings = max_position_embeddings
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.moe_intermediate_size = moe_intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.num_key_value_heads = num_key_value_heads
    self.n_shared_experts = n_shared_experts
    self.n_routed_experts = n_routed_experts
    self.routed_scaling_factor = routed_scaling_factor
    self.kv_lora_rank = kv_lora_rank
    self.q_lora_rank = q_lora_rank
    self.qk_rope_head_dim = qk_rope_head_dim
    self.v_head_dim = v_head_dim
    self.qk_nope_head_dim = qk_nope_head_dim
    self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    self.head_dim = qk_rope_head_dim
    self.n_group = n_group
    self.topk_group = topk_group
    self.num_experts_per_tok = num_experts_per_tok
    self.first_k_dense_replace = first_k_dense_replace
    self.norm_topk_prob = norm_topk_prob
    self.rope_interleave = rope_interleave
    self.hidden_act = hidden_act
    self.initializer_range = initializer_range
    self.rms_norm_eps = rms_norm_eps
    self.pretraining_tp = pretraining_tp
    self.use_cache = use_cache
    self.rope_theta = rope_theta
    self.rope_scaling = rope_scaling
    self.attention_bias = attention_bias
    self.attention_dropout = attention_dropout

    if self.rope_scaling is not None and "type" in self.rope_scaling:
      self.rope_scaling["rope_type"] = self.rope_scaling["type"]

    if self.rope_scaling is not None:
      for key in ["beta_fast", "beta_slow", "factor"]:
        if key in self.rope_scaling:
          self.rope_scaling[key] = float(self.rope_scaling[key])

    rope_config_validation(self)

    super().__init__(
      pad_token_id=pad_token_id,
      bos_token_id=bos_token_id,
      eos_token_id=eos_token_id,
      tie_word_embeddings=tie_word_embeddings,
      **kwargs,
    )


__all__ = ["DeepseekV3Config"]
