import dataclasses
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import dataclasses_json.cfg as json_cfg
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(frozen=True, eq=True)
class Metrics:
  """The metrics of a training run."""

  train_runtime: timedelta
  """The total time of the training run (including compilation)."""

  step_execution_time: timedelta | None
  """The average time to execute a training step."""

  # TODO(https://github.com/AI-Hypercomputer/torchprime/issues/67):
  # Add train_loss, compile_time, train_tokens_per_step, warm_train_tokens_per_second, etc.

  def __str__(self):
    s = ""
    for k, v in dataclasses.asdict(self).items():  # type: ignore
      value_str = str(v) if v is not None else "N/A"
      s += f"{k:20} = {value_str}\n"
    return s

  def save(self, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(self.to_json())  # type: ignore

  @staticmethod
  def load(path: Path) -> "Metrics":
    return Metrics.from_json(path.read_text())  # type: ignore


json_cfg.global_config.encoders[timedelta] = lambda obj: obj.total_seconds()
json_cfg.global_config.decoders[timedelta] = lambda obj: timedelta(seconds=obj)


class MetricsLogger:
  def __init__(self):
    self.start_time = time.time()
    self.step_execution_time = None

  def log_step_execution_time(self, step_execution_time: float):
    self.step_execution_time = step_execution_time

  def finalize(self) -> Metrics:
    return Metrics(
      train_runtime=timedelta(seconds=time.time() - self.start_time),
      step_execution_time=timedelta(seconds=self.step_execution_time)
      if self.step_execution_time
      else None,
    )
