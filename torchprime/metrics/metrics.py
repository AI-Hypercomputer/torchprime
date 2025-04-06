# from transformers.trainer_utils import TrainOutput
from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Metrics:
  """
  The metrics of a training run.
  """

  epoch: float
  """How many epochs have been completed. Includes fractions of epochs."""

  train_loss: float
  """The average training loss over the training run."""

  final_train_loss: float
  """The final training loss at the end of the training run."""

  train_runtime: float
  """The total runtime of the training run in seconds."""

  # TODO(https://github.com/AI-Hypercomputer/torchprime/issues/67):
  # Add compile_time, train_tokens_per_step, warm_train_tokens_per_second, etc.


class MetricsManager:
  pass

  def __init__(self):
    pass

  def on_step(self, step: int, loss: float):
    """
    Called at the end of each training step.
    """
    pass

  def finalize(self) -> Metrics:
    pass
