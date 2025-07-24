"""Trainer for Direct Preference Optimization (DPO)."""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import torch_xla
from omegaconf import DictConfig
from torch import nn

from torchprime.torch_xla_models.model import model_utils

from .sft_trainer import SFTTrainer

logger = logging.getLogger(__name__)


class DPOTrainer(SFTTrainer):
  """Trainer implementing a simple DPO objective."""

  def __init__(
    self,
    model: nn.Module,
    config: DictConfig,
    train_dataset,
  ) -> None:
    """Initialize the trainer and create the reference model.

    Args:
      model: The policy model to train.
      config: Hydra configuration specifying optimizer and model options.
      train_dataset: Dataset providing preference pairs.
    """
    self.beta = getattr(config.task, "beta", 0.1)
    super().__init__(model, config, train_dataset)

    dtype_name = config.get("torch_dtype", "bfloat16")
    model_dtype = getattr(torch, dtype_name)
    with model_utils.set_default_dtype(model_dtype), torch_xla.device():
      # The reference model shares the same architecture as the policy model
      # and is initialized from pretrained weights. It remains frozen during
      # training.
      self.ref_model = model_utils.initialize_model_class(config.model)
      if getattr(config.model, "pretrained_model", None):
        self.ref_model.from_pretrained(config.model.pretrained_model)
    # Keep the reference model on CPU unless needed to save TPU memory.
    self.ref_model.to("cpu")
    self.ref_model.eval()
    # Ensure the reference model does not receive gradient updates.
    for p in self.ref_model.parameters():
      p.requires_grad_(False)

  @contextmanager
  def _ref_model_on_device(self) -> Generator[None, None, None]:
    """Context manager to temporarily move the reference model to the XLA device."""
    self.ref_model.to(self.device)
    try:
      yield
    finally:
      self.ref_model.to("cpu")

  def _seq_log_prob(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute the log probability of a sequence.

    Args:
      logits: Model logits of shape ``[B, T, V]``.
      labels: Target token IDs of shape ``[B, T]`` with ``-100`` for padding.

    Returns:
      A tensor of shape ``[B]`` containing the summed log probabilities.
    """
    vocab = logits.size(-1)
    logits = logits[:, :-1].reshape(-1, vocab)
    labels = labels[:, 1:].reshape(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    labels_clipped = labels.clone()
    # Use a dummy index for padding positions so ``gather`` does not crash.
    labels_clipped[labels_clipped == -100] = 0
    token_log_probs = log_probs.gather(1, labels_clipped.unsqueeze(-1)).squeeze(-1)
    # Ignore padding tokens when summing probabilities.
    mask = labels != -100
    token_log_probs = token_log_probs * mask
    seq_log_probs = token_log_probs.view(labels.size()).sum(dim=1)
    return seq_log_probs

  @torch_xla.compile(full_graph=True)
  def train_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a single optimization step.

    The method computes the DPO loss between the current model and the
    reference model for a batch of preference pairs and updates the model
    parameters.

    Args:
      batch: A dictionary containing tokenized ``chosen`` and ``rejected``
        sequences.

    Returns:
      A tuple with the loss and gradient norm.
    """
    # Forward pass for the policy model on both the preferred and rejected responses.
    c_logits = self.model(
      input_ids=batch["chosen_input_ids"],
      attention_mask=batch["chosen_attention_mask"],
    )[0]
    r_logits = self.model(
      input_ids=batch["rejected_input_ids"],
      attention_mask=batch["rejected_attention_mask"],
    )[0]

    # Reference model forward pass is executed without gradient tracking. The
    # model is temporarily moved to the XLA device to save memory.
    with self._ref_model_on_device(), torch.no_grad():
      c_ref = self.ref_model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
      )[0]
      r_ref = self.ref_model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
      )[0]

    c_logp = self._seq_log_prob(c_logits, batch["chosen_labels"])
    r_logp = self._seq_log_prob(r_logits, batch["rejected_labels"])
    c_ref_logp = self._seq_log_prob(c_ref, batch["chosen_labels"])
    r_ref_logp = self._seq_log_prob(r_ref, batch["rejected_labels"])

    # DPO loss compares the advantage of the policy over the reference model
    # for the preferred vs. rejected responses.
    pi_logratios = c_logp - r_logp
    ref_logratios = c_ref_logp - r_ref_logp
    losses = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios))
    # Average over the batch to obtain the final loss.
    loss = losses.mean()
    loss.backward()
    grad_norm = self.clip_gradients()
    self.optimizer.step()
    self.lr_scheduler.step()
    # Clear gradients for the next iteration.
    self.model.zero_grad()
    return loss, grad_norm
