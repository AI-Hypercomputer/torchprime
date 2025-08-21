"""Utilities for comparing model implementations in tests."""

import torch
import torch_xla
from transformers.modeling_outputs import ModelOutput


def get_forward_pass_outputs(
  model: torch.nn.Module, **kwargs
) -> tuple[torch.Tensor, torch.Tensor | None]:
  """Gets the forward pass outputs (logits and loss) of a model.

  Args:
    model: The model to run the forward pass on.
    **kwargs: A dictionary of inputs for the model.

  Returns:
    A tuple containing the logits and loss. Loss may be None.
  """
  output = model(**kwargs)
  # Handle Hugging Face's ModelOutput class
  if isinstance(output, ModelOutput):
    loss = getattr(output, "loss", None)
    return output.logits, loss
  # Handle a raw (logits, loss) tuple for torchprime model
  elif isinstance(output, tuple) and len(output) == 2:
    return output
  else:
    raise ValueError(f"Unsupported model output format: {type(output)}")


def get_forward_and_backward_outputs(
  model: torch.nn.Module,
  **kwargs,
):
  """Runs a forward and backward pass, returning outputs and gradients."""
  model.zero_grad()
  logits, loss = get_forward_pass_outputs(model, **kwargs)

  if loss is None:
    raise ValueError("Cannot run backward pass without a loss value.")
  loss.backward()
  torch_xla.sync()

  return (logits, loss), model.named_parameters()
