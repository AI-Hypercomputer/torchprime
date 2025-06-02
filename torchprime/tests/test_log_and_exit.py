import logging
import unittest.mock

import torchprime.tools.log_and_exit


@unittest.mock.patch("sys.argv", ["log_and_exit.py", "profile_dir=/tmp", "output_dir=/tmp"])
def test_log_and_exit(caplog):
  """Test that main() logs expected key strings and returns 0."""
  # Act
  with caplog.at_level(logging.INFO):
    result = torchprime.tools.log_and_exit.main()

  # Assert
  print(caplog.text)
  assert result == 0
  assert "End of PyTorch Environment Logging" in caplog.text
  assert "PyTorch Environment Information" in caplog.text
  assert "Basic System Information" in caplog.text
  assert "ERROR" not in caplog.text

