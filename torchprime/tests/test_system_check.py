import logging
import unittest.mock

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

  # Verify that a log file was created in the temp directory
  log_files = list(output_dir.glob("log-*.log"))
  assert len(log_files) == 1, (
    f"Expected 1 log file, found {len(log_files)}: {log_files}"
  )
  log_file = log_files[0]
  assert log_file.exists()
  # Optionally verify the filename pattern
  assert log_file.name.startswith("log-")
  assert log_file.name.endswith(".log")
