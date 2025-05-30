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

  mfu: float | None
  """Model FLOPs Utilization."""

  tokens_per_second: float | None
  """The number of tokens processed per second during training."""

  num_steps: int | None
  """The number of steps performed during training."""

  # TODO(https://github.com/AI-Hypercomputer/torchprime/issues/67):
  # Add train_loss, compile_time, train_tokens_per_step, warm_train_tokens_per_second, etc.
  # Document them in docs/metrics.md too.

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
    self.train_runtime = None
    self.start_time = time.time()
    self.step_execution_time = None
    self.mfu = None
    self.tokens_per_second = None
    self.num_steps = None

  def log_step_execution_time(self, step_execution_time: float):
    self.step_execution_time = step_execution_time

  def log_mfu(self, mfu: float):
    self.mfu = mfu

  def log_tokens_per_second(self, tokens_per_second: float):
    self.tokens_per_second = tokens_per_second

  def log_num_steps(self, num_steps: int):
    self.num_steps = num_steps

  def finalize(self) -> Metrics:
    self.train_runtime = timedelta(seconds=time.time() - self.start_time)
    return Metrics(
      train_runtime=timedelta(seconds=time.time() - self.start_time),
      step_execution_time=timedelta(seconds=self.step_execution_time)
      if self.step_execution_time
      else None,
      mfu=self.mfu,
      tokens_per_second=self.tokens_per_second,
      num_steps=self.num_steps,
    )
