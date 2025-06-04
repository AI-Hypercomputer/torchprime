import logging
import unittest.mock

import pytest

import torchprime.tools.system_check


def test_system_check(caplog, tmp_path):
  """Test that main() logs expected key strings and returns 0 with dynamic temp dirs."""
  # Create subdirectories in the temp path
  output_dir = tmp_path / "output"
  profile_dir = tmp_path / "profile"
  output_dir.mkdir()
  profile_dir.mkdir()

  # Patch sys.argv with dynamic temp directories
  with (
    unittest.mock.patch(
      "sys.argv",
      ["log_and_exit.py", f"output_dir={output_dir}", f"profile_dir={profile_dir}"],
    ),
    caplog.at_level(logging.INFO),
  ):
    # Act
    result = torchprime.tools.system_check.main()

  # Assert
  assert result == 0
  assert "End of PyTorch Environment Logging" in caplog.text
  assert "PyTorch Environment Information" in caplog.text
  assert "Basic System Information" in caplog.text
  assert "ERROR" not in caplog.text

  # Verify log file was created in the temp directory
  log_file = output_dir / "log.log"
  assert log_file.exists()
