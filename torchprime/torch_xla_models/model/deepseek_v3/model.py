"""Minimal DeepSeek V3 model for SFT.

This module reuses the LLaMA architecture to provide a DeepSeek V3
implementation suitable for supervised fine-tuning (SFT) workloads.
"""

from torchprime.torch_xla_models.model.llama.model import (
    LlamaForCausalLM,
    LlamaModel,
)


class DeepseekModel(LlamaModel):
  """Alias of :class:`LlamaModel` used for DeepSeek V3."""


class DeepseekForCausalLM(LlamaForCausalLM):
  """DeepSeek V3 model for causal language modeling."""

  def __init__(self, config):
    super().__init__(config)
