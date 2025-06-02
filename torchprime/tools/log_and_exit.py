"""Utility to log environment information and exit gracefully."""

import datetime
import logging
import os
import platform
import sys

logging.basicConfig(
  format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
  datefmt="%m/%d/%Y %H:%M:%S",
  level=logging.INFO,
)

logger = logging.getLogger(__name__)


def log_basic_system_info() -> None:
  """Logs basic system and Python information."""
  logger.info("--- Basic System Information ---")
  logger.info(f"Timestamp: {datetime.datetime.now().isoformat()}")
  logger.info(f"Hostname: {platform.node()}")
  logger.info(f"Platform: {platform.platform()}")
  logger.info(f"Machine: {platform.machine()}")
  logger.info(f"Processor: {platform.processor()}")
  logger.info(f"Architecture: {platform.architecture()[0]}")
  logger.info(f"System: {platform.system()} {platform.release()}")
  logger.info(f"Version: {platform.version()}")
  logger.info(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
  logger.info(f"Python Version: {sys.version.replace(chr(10), ' ')}")  # Remove newlines
  logger.info(f"Python Executable: {sys.executable}")


def log_args() -> None:
  """Logs command line arguments."""
  logger.info("--- Command Line Arguments ---")
  if len(sys.argv) == 1:
    logger.info("No command line arguments provided.")
  else:
    for i, arg in enumerate(sys.argv):
      logger.info(f"Argument {i}: {arg}")
  logger.info(f"Total arguments: {len(sys.argv)}")


def log_all_env_variables() -> None:
  """Logs all environment variables."""
  logger.info("--- All Environment Variables ---")
  if not os.environ:
    logger.info("No environment variables found.")
    return

  for key, value in sorted(os.environ.items()):
    logger.info(f"{key}={value}")
  logger.info(f"Logged {len(os.environ)} environment variable(s).")


def log_pytorch_info() -> None:
  """Logs PyTorch information, version, and performs checks."""
  logger.info("--- PyTorch Information ---")
  try:
    import torch
  except ImportError:
    logger.warning("torch not found. Exiting without further checks.")
    return

  logger.info(f"torch imported successfully. torch version: {torch.__version__}")

  # CPU tensor addition
  try:
    a_cpu = torch.tensor([1.0, 2.0, 3.0])
    b_cpu = torch.tensor([4.0, 5.0, 6.0])
    c_cpu = a_cpu + b_cpu
    logger.info(f"torch CPU tensor addition (a+b): {a_cpu} + {b_cpu} = {c_cpu}")
  except Exception as e:
    logger.error(f"Error during torch CPU tensor operation: {e}")

  # Check CUDA availability
  try:
    cuda_available = torch.cuda.is_available()
    logger.info(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
      logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
      logger.info(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
      logger.info(
        f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A (No CUDA devices)'}"
      )
  except Exception as e:
    logger.error(f"Error checking CUDA availability: {e}")

  # Check MPS availability (for Apple Silicon)
  try:
    # Check if MPS is available and built with PyTorch
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
      mps_available = True
      # Further check if a device can be allocated. is_built() is not always enough.
      try:
        torch.tensor([1], device="mps")
        mps_functional = True
      except Exception:
        mps_functional = False
      logger.info(
        f"torch.backends.mps.is_available(): {mps_available} (Functional: {mps_functional})"
      )
    elif hasattr(torch.backends, "mps"):  # mps backend exists but not available
      logger.info(
        f"torch.backends.mps.is_available(): False (torch.backends.mps.is_built(): {torch.backends.mps.is_built()})"
      )
    else:  # mps backend does not exist
      logger.info("torch.backends.mps not available in this PyTorch build.")
  except Exception as e:
    logger.error(f"Error checking MPS availability: {e}")

  logger.info("--- PyTorch/XLA Information ---")
  try:
    import torch_xla
  except ImportError:
    logger.warning("torch_xla not found. Exiting without further checks.")
    return

  try:
    logger.info("torch_xla imported successfully.")
    logger.info(f"torch_xla version: {torch_xla.__version__}")

    device = torch.device("xla")

    a_xla = torch.tensor([10.0, 20.0], device=device)
    b_xla = torch.tensor([30.0, 40.0], device=device)
    c_xla = a_xla + b_xla
    # xm.mark_step() # Often needed in XLA training loops, not strictly for a single op
    logger.info(
      f"torch_xla tensor addition on {device} (a+b): {a_xla} + {b_xla} = {c_xla}"
    )
  except Exception as e:
    logger.error(f"Error during torch_xla operations: {e}")
    return


def main() -> int:
  # Don't use a fancy arg parser here. We are just pulling 2 args.
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
    log_file = os.path.join(output_dir, "log.log")
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
  log_pytorch_info()
  logger.info("----------------------------------------------------------------------")

  logger.info("======================================================================")
  logger.info("                  End of PyTorch Environment Logging                  ")
  logger.info("======================================================================")

  return 0


if __name__ == "__main__":
  sys.exit(main())
