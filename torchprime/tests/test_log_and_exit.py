import logging
import unittest.mock

import pytest

import torchprime.tools.log_and_exit


def test_log_and_exit_with_dynamic_tempdir(caplog, tmp_path):
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
    result = torchprime.tools.log_and_exit.main()

  # Assert
  assert result == 0
  assert "End of PyTorch Environment Logging" in caplog.text
  assert "PyTorch Environment Information" in caplog.text
  assert "Basic System Information" in caplog.text
  assert "ERROR" not in caplog.text

  # Verify log file was created in the temp directory
  log_file = output_dir / "log.log"
  assert log_file.exists()


@pytest.fixture
def temp_dirs(tmp_path):
  """Fixture that creates output and profile temp directories."""
  output_dir = tmp_path / "output"
  profile_dir = tmp_path / "profile"
  output_dir.mkdir()
  profile_dir.mkdir()
  return {"output_dir": output_dir, "profile_dir": profile_dir}


def test_log_and_exit_with_fixture(caplog, temp_dirs):
  """Test using a custom fixture for temp directories."""
  with (
    unittest.mock.patch(
      "sys.argv",
      [
        "log_and_exit.py",
        f"output_dir={temp_dirs['output_dir']}",
        f"profile_dir={temp_dirs['profile_dir']}",
      ],
    ),
    caplog.at_level(logging.INFO),
  ):
    result = torchprime.tools.log_and_exit.main()

  assert result == 0
  assert "End of PyTorch Environment Logging" in caplog.text
  assert "PyTorch Environment Information" in caplog.text
  assert "Basic System Information" in caplog.text
  assert "ERROR" not in caplog.text
