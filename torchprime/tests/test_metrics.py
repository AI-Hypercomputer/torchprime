from datetime import timedelta

from torchprime.metrics.metrics import Metrics


def test_print_metrics():
  """Test the string representation of the Metrics class."""

  # All fields are set.
  metrics = Metrics(
    train_runtime=timedelta(seconds=3600.0),
    step_execution_time=timedelta(seconds=63.1),
  )
  expected_str = """
train_runtime        = 1:00:00
step_execution_time  = 0:01:03.100000
""".lstrip()
  assert str(metrics) == expected_str

  # Missing step_execution_time.
  metrics = Metrics(
    train_runtime=timedelta(seconds=3600.0),
    step_execution_time=None,
  )
  expected_str = """
train_runtime        = 1:00:00
step_execution_time  = N/A
""".lstrip()
  assert str(metrics) == expected_str


def test_metrics_to_json():
  metrics = Metrics(
    train_runtime=timedelta(seconds=1),
    step_execution_time=timedelta(seconds=1),
  )
  json_str = metrics.to_json()  # type: ignore
  metrics_back = Metrics.from_json(json_str)  # type: ignore
  assert metrics == metrics_back
