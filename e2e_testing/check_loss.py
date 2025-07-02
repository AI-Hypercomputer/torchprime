#!/usr/bin/env python3
"""Utility to validate the final loss reported during training.

Example usage: python e2e_testing/check_loss.py <log_file> <target_loss> <tolerance>
"""

import re
import sys

LOSS_PATTERN = re.compile(r"loss:\s*([0-9]+\.?[0-9]*)")


def check_loss(file_path: str, target_loss: float, tolerance: float) -> int:
  """Check that the loss in ``file_path`` is within ``tolerance`` of ``target_loss``.

  Args:
    file_path: Path to the log file to parse.
    target_loss: Expected final loss value.
    tolerance: Allowed deviation from ``target_loss``.

  Returns:
    ``0`` if the check passes, ``1`` otherwise.
  """

  try:
    with open(file_path) as f:
      log_data = f.read()
  except Exception as e:  # pylint: disable=broad-except
    print(f"Error reading log file {file_path}: {e}")
    return 1

  matches = LOSS_PATTERN.findall(log_data)
  if not matches:
    print("Error: No loss value found in logs")
    return 1

  last_loss = float(matches[-1])
  print(f"Last loss found: {last_loss}")

  if abs(last_loss - target_loss) > tolerance:
    print(
      f"Error: loss {last_loss:.4f} not within ±{tolerance} of target {target_loss:.4f}"
    )
    return 1

  print("Loss check passed.")
  return 0


if __name__ == "__main__":
  if len(sys.argv) != 4:
    print("Usage: check_loss.py <log_file> <target_loss> <tolerance>")
    sys.exit(1)

  sys.exit(check_loss(sys.argv[1], float(sys.argv[2]), float(sys.argv[3])))
