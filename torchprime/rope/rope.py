"""
Rotary Positional Embeddings (RoPE) implementation.
Reference: https://github.com/adalkiran/llama-nuts-and-bolts/blob/main/docs/10-ROPE-ROTARY-POSITIONAL-EMBEDDINGS.md
"""

import math
from dataclasses import dataclass

import torch
from omegaconf import DictConfig


@dataclass(kw_only=True)
class RopeScaling:
  """
  RoPE scaling parameters. The defaults are what was selected in Llama 3.1.
  """

  factor: float = 8.0
  low_freq_factor: float = 1.0
  high_freq_factor: float = 4.0
  original_context_len: int = 8192


def default_rope_frequencies(
  head_dim: int,
  theta: float = 10000.0,
) -> torch.Tensor:
  """
  Computes the original RoPE frequencies in e.g. Llama 2.
  Args:
      head_dim: the size of a single attention head.
      theta: a hyperparameter controlling how fast the embeddings rotate.
  Returns:
      The frequencies for the RoPE embeddings.
  """
  return 1.0 / (
    theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
  )


def llama3_rope_frequencies(
  head_dim: int,
  theta: float = 10000.0,
  scaling: RopeScaling | None = None,
) -> torch.Tensor:
  """
  Computes Llama 3 and 3.1 RoPE frequencies. In Llama 3.1, RoPE frequencies
  may be scaled and interpolated as we move beyond the original context length.
  """
  freqs = default_rope_frequencies(head_dim=head_dim, theta=theta)
  if scaling is None:
    return freqs

  low_freq_wavelen = scaling.original_context_len / scaling.low_freq_factor
  high_freq_wavelen = scaling.original_context_len / scaling.high_freq_factor

  assert low_freq_wavelen >= high_freq_wavelen, (
    f"low_freq_wavelen {low_freq_wavelen} must be greater or equal to "
    f"high_freq_wavelen {high_freq_wavelen}"
  )

  wavelen = 2 * math.pi / freqs
  # wavelen < high_freq_wavelen: do nothing
  # wavelen > low_freq_wavelen: divide by factor
  freqs = torch.where(wavelen > low_freq_wavelen, freqs / scaling.factor, freqs)
  # otherwise: interpolate between the two, using a smooth factor
  smooth_factor = (scaling.original_context_len / wavelen - scaling.low_freq_factor) / (
    scaling.high_freq_factor - scaling.low_freq_factor
  )
  smoothed_freqs = (1 - smooth_factor) * freqs / scaling.factor + smooth_factor * freqs
  is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
  freqs = torch.where(is_medium_freq, smoothed_freqs, freqs)

  return freqs


def deepseek_v3_rope_init_fn(config: DictConfig) -> tuple["torch.Tensor", float]:
  """
  copied from HF implementation `_compute_yarn_parameters` function, from
  https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L197C5-L197C29

  Computes the inverse frequencies with NTK scaling. Please refer to the
  [original paper](https://huggingface.co/papers/2309.00071)
  Args:
      config ([`~transformers.PretrainedConfig`]):
          The model configuration.
  Returns:
      Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
      post-processing scaling factor applied to the computed cos/sin.
  """

  assert hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, DictConfig)
  assert config.rope_scaling.get("rope_type", config.rope_scaling.get("type")) == "yarn"
  base = config.rope_theta
  partial_rotary_factor = (
    config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
  )
  head_dim = getattr(
    config, "head_dim", config.hidden_size // config.num_attention_heads
  )
  dim = int(head_dim * partial_rotary_factor)
  factor = config.rope_scaling["factor"]
  attention_factor = config.rope_scaling.get("attention_factor")
  mscale = config.rope_scaling.get("mscale")
  mscale_all_dim = config.rope_scaling.get("mscale_all_dim")

  # NOTE: DeekSeek-V3 (and potentially other models) modify `max_position_embeddings` and have a
  # `original_max_position_embeddings` field containing the pretrained value. They use the ratio between these two
  # values to compute the default attention scaling factor, instead of using `factor`.
  if "original_max_position_embeddings" in config.rope_scaling:
    original_max_position_embeddings = config.rope_scaling[
      "original_max_position_embeddings"
    ]
    factor = config.max_position_embeddings / original_max_position_embeddings
  else:
    original_max_position_embeddings = config.max_position_embeddings

  def get_mscale(scale, mscale=1):
    if scale <= 1:
      return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

  # Sets the attention factor as suggested in the paper
  if attention_factor is None:
    if mscale and mscale_all_dim:
      attention_factor = float(
        get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
      )
    else:
      attention_factor = get_mscale(factor)

  # Optional config options
  # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
  beta_fast = config.rope_scaling.get("beta_fast") or 32
  beta_slow = config.rope_scaling.get("beta_slow") or 1

  # Compute the inverse frequencies
  def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    """Inverse dimension formula to find the dimension based on the number of rotations"""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
      2 * math.log(base)
    )

  def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
    """Find dimension range bounds based on rotations"""
    low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)

  def linear_ramp_factor(min, max, dim):
    if min == max:
      max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

  # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
  # to expand the possible context length. In other words, interpolation = apply scaling factor.
  pos_freqs = base ** (torch.arange(0, dim, 2).to(dtype=torch.float) / dim)
  inv_freq_extrapolation = 1.0 / pos_freqs
  inv_freq_interpolation = 1.0 / (factor * pos_freqs)

  low, high = find_correction_range(
    beta_fast, beta_slow, dim, base, original_max_position_embeddings
  )

  # Get n-dimensional rotational scaling corrected for extrapolation
  inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(
    dtype=torch.float
  )
  inv_freq = (
    inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
    + inv_freq_extrapolation * inv_freq_extrapolation_factor
  )
  return inv_freq, attention_factor
