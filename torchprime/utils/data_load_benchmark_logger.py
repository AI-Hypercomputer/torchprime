import csv
from pathlib import Path
from typing import Any


class DataLoadBenchmarkLogger:
  """A simple logger for writing data loading benchmark data to a CSV file."""

  def __init__(self, output_dir: str, filename: str, fieldnames: list[str]):
    """Initializes the logger.

    Args:
      output_dir: The directory where the log file will be saved.
      filename: The name of the CSV file.
      fieldnames: The list of column names for the CSV file.
    """
    self.output_path = Path(output_dir) / filename
    self.fieldnames = fieldnames
    # Ensure the output directory exists.
    self.output_path.parent.mkdir(parents=True, exist_ok=True)

  def log_step(self, **kwargs: Any):
    """Logs a single step of benchmark data."""
    file_exists = self.output_path.exists()

    with self.output_path.open("a", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=self.fieldnames)
      if not file_exists or f.tell() == 0:
        writer.writeheader()
      writer.writerow(kwargs)
