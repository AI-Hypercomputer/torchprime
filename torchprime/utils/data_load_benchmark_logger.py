import csv
from pathlib import Path


class DataLoadBenchmarkLogger:
  """A simple logger for writing data loading benchmark data to a CSV file."""

  def __init__(self, output_dir: str, filename: str):
    """Initializes the logger.

    Args:
      output_dir: The directory where the log file will be saved.
      filename: The name of the CSV file.
    """
    self.output_path = Path(output_dir) / filename
    self.file = None
    self.writer = None

  def log_step(self, **kwargs):
    """Logs a single step of benchmark data.

    The first call to this method determines the CSV header from the keys
    of the provided keyword arguments.
    """
    with open(self.output_path, "a", newline="") as csvfile:
      fieldnames = list(kwargs.keys())
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

      # Write header if the file is empty
      if csvfile.tell() == 0:
        writer.writeheader()

    self.writer.writerow(kwargs)

  def writerow(self, row):
    with open(self.output_path, "a", newline="") as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
      writer.writerow(row)

  def __del__(self):
    pass
