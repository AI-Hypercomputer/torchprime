"""Utility to log system info and check PyTorch environment."""

import datetime
import logging
import os
import platform
import sys
from importlib.metadata import distributions

logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
  datefmt="%m/%d/%Y %H:%M:%S",
  level=logging.INFO,
)

logger = logging.getLogger(__name__)


def log_basic_system_info() -> None:
  """Logs basic system and Python information."""
  logger.info("--- Basic System Information ---")
  logger.info("Timestamp: %s", datetime.datetime.now().isoformat())
  logger.info("Hostname: %s", platform.node())
  logger.info("Platform: %s", platform.platform())
  logger.info("Machine: %s", platform.machine())
  logger.info("Processor: %s", platform.processor())
  logger.info("Architecture: %s", platform.architecture()[0])
  logger.info("System: %s %s", platform.system(), platform.release())
  logger.info("Version: %s", platform.version())
  logger.info(
    "OS: %s %s (%s)", platform.system(), platform.release(), platform.version()
  )
  logger.info(
    "Python Version: %s", sys.version.replace(chr(10), " ")
  )  # Remove newlines
  logger.info("Python Executable: %s", sys.executable)


def log_args() -> None:
  """Logs command line arguments."""
  logger.info("--- Command Line Arguments ---")
  if len(sys.argv) == 1:
    logger.info("No command line arguments provided.")
  else:
    for i, arg in enumerate(sys.argv):
      logger.info("Argument %d: %s", i, arg)
  logger.info("Total arguments: %d", len(sys.argv))


def log_all_env_variables() -> None:
  """Logs all environment variables."""
  logger.info("--- All Environment Variables ---")
  if not os.environ:
    logger.info("No environment variables found.")
    return

  for key, value in sorted(os.environ.items()):
    logger.info("%s=%s", key, value)
  logger.info("Logged %d environment variable(s).", len(os.environ))


def log_python_package_info() -> None:
  logger.info("--- Python Package Information ---")
  dists = list(sorted(distributions(), key=lambda d: d.metadata["Name"].lower()))
  for dist in dists:
    logger.info("%s==%s", dist.metadata["Name"], dist.version)

  logger.info("Logged %d Python package(s).", len(dists))


def log_pytorch_info() -> None:
  """Logs PyTorch information, version, and performs checks."""
  logger.info("--- PyTorch Information ---")
  try:
    import torch
  except ImportError:
    logger.warning("torch not found. Exiting without further checks.")
    return

  logger.info("torch imported successfully. torch version: %s", torch.__version__)

  # CPU tensor addition
  try:
    a_cpu = torch.tensor([1.0, 2.0, 3.0])
    b_cpu = torch.tensor([4.0, 5.0, 6.0])
    c_cpu = a_cpu + b_cpu
    logger.info("torch CPU tensor addition (a+b): %s + %s = %s", a_cpu, b_cpu, c_cpu)
  except Exception as e:
    logger.error("Error during torch CPU tensor operation: %s", e)

  logger.info("--- PyTorch/XLA Information ---")
  try:
    import torch_xla
  except ImportError:
    logger.warning("torch_xla not found. Exiting without further checks.")
    return

  try:
    logger.info("torch_xla imported successfully.")
    logger.info("torch_xla version: %s", torch_xla.__version__)

    device = torch.device("xla")

    a_xla = torch.tensor([10.0, 20.0], device=device)
    b_xla = torch.tensor([30.0, 40.0], device=device)
    c_xla = a_xla + b_xla
    # xm.mark_step() # Often needed in XLA training loops, not strictly for a single op
    logger.info(
      "torch_xla tensor addition on %s (a+b): %s + %s = %s", device, a_xla, b_xla, c_xla
    )
  except Exception as e:
    logger.error("Error during torch_xla operations: %s", e)
    return


def main() -> int:
  # Don't use a fancy arg parser here. We are just pulling 2 args.
  output_dir = None
  for arg in sys.argv:
    if arg.startswith("output_dir="):
      output_dir = arg.split("=", 1)[1]
      break
  else:
    logger.error(
      "No output_dir argument provided. Not attaching logger to output directory."
    )

  if output_dir:
    # Attach logger to the output directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(output_dir, f"log-{timestamp}.log")
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Created output directory: %s", output_dir)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
      "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Attached logger to file: %s", log_file)

  logger.info("======================================================================")
  logger.info("                       PyTorch Environment Information                ")
  logger.info("======================================================================")

  log_args()
  logger.info("----------------------------------------------------------------------")
  log_basic_system_info()
  logger.info("----------------------------------------------------------------------")
  log_all_env_variables()
  logger.info("----------------------------------------------------------------------")
  log_python_package_info()
  logger.info("----------------------------------------------------------------------")
  log_pytorch_info()
  logger.info("----------------------------------------------------------------------")

  logger.info("======================================================================")
  logger.info("                  End of PyTorch Environment Logging                  ")
  logger.info("======================================================================")

  return 0


if __name__ == "__main__":
  sys.exit(main())
