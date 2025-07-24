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
    self.output_path = Path(output_dir)
    self.output_path.mkdir(parents=True, exist_ok=True)
    self.output_path /= filename
    # Open in append mode to support resuming training.
    self.file = open(self.output_path, "a", newline="")
    self.writer = None
    # Check if we need to write a header. If file is not empty, header is assumed to exist.
    self.header_written = self.file.tell() > 0

  def log_step(self, **kwargs):
    """Logs a single step of benchmark data.

    The first call to this method also writes the CSV header.
    """
    if self.file is None:
      # Logger has been closed.
      return

    if self.writer is None:
      fieldnames = list(kwargs.keys())
      self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
      if not self.header_written:
        self.writer.writeheader()
        self.header_written = True

    self.writer.writerow(kwargs)
    self.file.flush()

  def close(self):
    """Closes the underlying file."""
    if self.file:
      self.file.close()
      self.file = None
      self.writer = None

  def __del__(self):
    self.close()
