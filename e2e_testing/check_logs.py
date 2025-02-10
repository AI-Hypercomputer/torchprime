#!/usr/bin/env python3
import re
import sys


def check_logs(file_path):
  try:
    with open(file_path) as f:
      log_data = f.read()
  except Exception as e:
    print(f"Error reading log file {file_path}: {e}")
    return 1

  # Validate that the log contains the expected patterns.
  if not re.search(r"Finished training run", log_data):
    print("Error: 'Finished training run' not found in logs")
    return 1

  step_duration = re.search(r"Step duration:.*s", log_data)
  if not step_duration:
    print("Error: 'Step duration' not found in logs")
    return 1

  print(step_duration.group())
  print("Logs check passed.")
  return 0


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: check_logs.py <log_file>")
    sys.exit(1)
  sys.exit(check_logs(sys.argv[1]))
