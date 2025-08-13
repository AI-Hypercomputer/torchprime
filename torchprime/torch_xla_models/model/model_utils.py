"""Utility function(s) for model initialization."""

from __future__ import annotations

import importlib
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import huggingface_hub
import safetensors
import torch
import torch.distributed.checkpoint as dist_cp
import torch_xla.experimental.distributed_checkpoint as xc
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

HF_MODEL_CONFIG_FILES = [
  "config.json",
  "generation_config.json",
]


def load_safetensors_to_state_dict(model_dir: str) -> dict:
  """Load a model state dict from safetensors, supporting both sharded and single-file formats.

  This function loads model weights from the specified directory. It supports both
  sharded (`model.safetensors.index.json`) and single-file (`model.safetensors`) formats.

  Args:
      model_dir: Path to the directory containing the model files.

  Returns:
      dict: A state dictionary containing the model's parameters.

  Raises:
      FileNotFoundError: If neither the sharded nor single-file safetensors are found.
  """

  state_dict = {}
  index_file = os.path.join(model_dir, "model.safetensors.index.json")
  single_file = os.path.join(model_dir, "model.safetensors")

  if os.path.exists(index_file):
    # Load sharded safetensors
    with open(index_file) as f:
      index = json.load(f)
    weight_map = index["weight_map"]
    for filename in set(weight_map.values()):
      path = os.path.join(model_dir, filename)
      with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118
          state_dict[key] = f.get_tensor(key)
  elif os.path.exists(single_file):
    # Load single safetensor file
    state_dict = safetensors.torch.load_file(single_file)
  else:
    raise FileNotFoundError(
      f"No safetensors found in {model_dir}. Expected 'model.safetensors' or 'model.safetensors.index.json'."
    )

  return state_dict


def save_sharded_safetensors_by_layer(
  state_dict: dict[str, torch.Tensor],
  save_dir: str | os.PathLike,
  *,
  max_workers: int = 24,
  tmp_dir: str | os.PathLike | None = None,
):
  """Save a model state dict to sharded safetensors by layer prefix.

  This function saves the model's state dictionary into separate sharded files,
  grouped by the top-level layer prefix. It also creates an index file
  (`model.safetensors.index.json`) mapping each parameter to its corresponding shard.

  Args:
      state_dict (dict): The model's state dictionary to be saved.
      save_dir (str): Directory where the sharded safetensors and index file will be saved.
      max_workers: Parallel writer threads.  24 saturates v6e-4 dual NVMe; tune as needed.
      tmp_dir: If given, write shards to this *local* directory first, then
        copy the results to ``save_dir``.  Handy when `save_dir` is a
        Cloud-Storage mount and you want full NVMe speed.
  """
  save_dir = Path(save_dir)
  if tmp_dir:
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    work_dir = tmp_dir
  else:
    save_dir.mkdir(parents=True, exist_ok=True)
    work_dir = save_dir
  work_dir.mkdir(parents=True, exist_ok=True)

  grouped: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
  sizes: dict[str, int] = {}
  for k, v in state_dict.items():
    p = get_param_group_key(k)
    grouped[p][k] = v
    sizes[p] = sizes.get(p, 0) + v.numel() * v.element_size()

  def _write_one(item: tuple[str, dict[str, torch.Tensor]]) -> dict[str, str]:
    prefix, group = item
    fname = f"{prefix}.safetensors"
    safetensors.torch.save_file(group, str(work_dir / fname))
    # strip FSDP suffix for HF compatibility
    return {k.replace("._orig_mod", ""): fname for k in group}

  # sort largest → smallest so threads finish together
  items = sorted(grouped.items(), key=lambda kv: sizes[kv[0]], reverse=True)

  weight_map: dict[str, str] = {}
  with ThreadPoolExecutor(max_workers=max_workers) as pool:
    for mapping in pool.map(_write_one, items):
      weight_map.update(mapping)

  # ---------- dump index --------------------------------------------
  (work_dir / "model.safetensors.index.json").write_text(
    json.dumps({"weight_map": weight_map}, indent=2)
  )

  # ---------- sync to remote if needed ------------------------------
  maybe_move_to_mounted_gcs(tmp_dir, save_dir)


def initialize_model_class(model_config):
  """Import and initialize model_class specified by the config."""
  full_model_class_string = model_config.model_class
  module_name, model_class_name = full_model_class_string.rsplit(".", 1)
  module = None

  for candidate_module_name in [f"model.{module_name}", module_name]:
    # use full import path to avoid issues with relative imports
    full_module_name = f"torchprime.torch_xla_models.{candidate_module_name}"
    try:
      module = importlib.import_module(full_module_name)
      break
    except ModuleNotFoundError:
      module = None

  if module is None:
    print(f"Error: Failed to import module '{module_name}' or 'model.{module_name}'")
    sys.exit(1)

  if not hasattr(module, model_class_name):
    print(f"Error: Class '{model_class_name}' not found in module '{module.__name__}'")
    sys.exit(1)

  model_class = getattr(module, model_class_name)
  return model_class(model_config)


@contextmanager
def set_default_dtype(dtype: torch.dtype):
  """Temporarily sets the default torch dtype within a context.

  This context manager sets the PyTorch default floating point dtype
  (e.g., `torch.bfloat16`) for the duration of the context
  and restores the original dtype afterward.

  Example:
      ```python
      with set_default_dtype(torch.bfloat16):
          model = MyModel()  # initialized with bfloat16 weights
      ```

  Args:
      dtype: The dtype to set as default within the context.
  """
  previous_dtype = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    yield
  finally:
    torch.set_default_dtype(previous_dtype)


def extract_model_size_from_model_name(model_name: str) -> int | float:
  """Extract the model size in billions from a model name string.

  Args:
      model_name (str): The model name string, e.g., "llama-3-8b.yaml".

  Returns:
      Union[int, float]: The model size in billions, or -1 if not found.
  """
  match = re.search(r"(\d+(?:\.\d+)?)b", model_name.lower())
  if match:
    size_str = match.group(1)
    try:
      size = float(size_str)
      return int(size) if size.is_integer() else size
    except ValueError:
      return -1
  return -1


def log_parameter_breakdown(model: torch.nn.Module, logger: logging.Logger) -> None:
  """Logs the number of parameters in different components of the model.

  Args:
      model: The PyTorch model.
      logger: A logger instance to write the output to.
  """
  total_params = sum(p.numel() for p in model.parameters())
  logger.info("Model total size: {} parameters".format(f"{total_params:,}"))

  param_groups = {
    "mlp": 0,
    "attention": 0,
    "embedding": 0,
    "lm_head": 0,
    "norm": 0,
    "other": 0,
  }

  for name, param in model.named_parameters():
    if "mlp" in name:
      param_groups["mlp"] += param.numel()
    elif "self_attn" in name or "attention" in name:
      param_groups["attention"] += param.numel()
    elif "embed" in name:
      param_groups["embedding"] += param.numel()
    elif "lm_head" in name:
      param_groups["lm_head"] += param.numel()
    elif "norm" in name or "layernorm" in name:
      param_groups["norm"] += param.numel()
    else:
      param_groups["other"] += param.numel()

  for k, v in param_groups.items():
    percentage = (v / total_params) * 100
    logger.info("  {:10s}: {} params ({:.2f}%)".format(k, f"{v:,}", percentage))


def get_param_group_key(param_name: str) -> str:
  """Return a deterministic *shard key* for any parameter name.

  Heuristics (in priority order):

  1. Strip a leading ``module.`` (used by `nn.DataParallel` / `FSDP` wraps).
  2. Transformer-style: ``{model,encoder,decoder}.layers.<idx>.…`` ➜ ``layers_<idx>``.
  3. Generic block pattern ``<block>.<idx>.…`` (e.g. ``layer1.0.*``,
     ``down_blocks.3.*``) ➜ ``<block>_<idx>``.
  4. Fallback to the first path component (``lm_head``, ``bias`` …).

  The returned string is short (good for filenames) and stable across runs.
  """
  parts = param_name.split(".")

  # (1) Strip wrapper prefixes inserted by some wrappers
  if parts[0] == "module":
    parts = parts[1:]

  # (2) Transformer blocks (model.layers.N / encoder.layers.N / decoder.layers.N)
  if (
    len(parts) >= 3
    and parts[0] in {"model", "encoder", "decoder"}
    and parts[1] == "layers"
    and parts[2].isdigit()
  ):
    return f"layers_{parts[2]}"

  # (3) Generic "<block>.<idx>.*" – covers ResNet, UNet, Swin, ConvNeXt …
  if len(parts) >= 2 and parts[1].isdigit():
    return f"{parts[0]}_{parts[1]}"

  # (4) Last-chance fallback
  return parts[0]


def save_distributed_checkpoint(model: torch.nn.Module, save_dir: Path) -> None:
  """Save the model state using torch.distributed.checkpoint (DCP).

  Args:
    model: The model whose state_dict should be saved.
    save_dir: Directory where the distributed checkpoint will be written.
  """
  state_dict = {"model": model.state_dict()}
  dist_cp.save(
    state_dict=state_dict,
    storage_writer=dist_cp.FileSystemWriter(
      str(save_dir), thread_count=max(2, min(8, mp.cpu_count()))
    ),
    planner=xc.SPMDSavePlanner(),
  )
  logger.info("DCP checkpoint written to %s", save_dir)


def convert_to_safetensors_on_cpu(model: torch.nn.Module, save_dir: Path) -> None:
  """Reload checkpoint on CPU and export sharded safetensors files.

  Assumes the checkpoint has already been written. This method should only be run on Rank 0.
  Exaple usage:
  ```python
  if xr.process_index() == 0:
    model_utils.convert_to_safetensors_on_cpu(...)
  ```

  Args:
    model: The model used to construct the CPU placeholder state_dict.
    save_dir: Directory where the original checkpoint is saved.
              Safetensors files and index will also be written here.
  """
  logger.info("Reloading checkpoint for safetensors export …")

  model_sd = model.state_dict()
  reload_sd = {
    "model": {
      name: torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
      for name, tensor in model_sd.items()
    }
  }

  dist_cp.load(
    state_dict=reload_sd,
    storage_reader=dist_cp.FileSystemReader(str(save_dir)),
    planner=xc.SPMDLoadPlanner(),
  )
  logger.info("Checkpoint fully materialised on CPU")

  # delete the checkpoint dist_cp files `*.distcp` to save disk space
  for file in save_dir.glob("*.distcp"):
    try:
      file.unlink()
    except OSError as e:
      logger.warning("Failed to delete %s: %s", file, str(e))

  cpu_state = {k.replace("._orig_mod", ""): v for k, v in reload_sd["model"].items()}

  try:
    tmp_dir = tempfile.mkdtemp(dir="/mnt/localssd")
    logger.info("Using local SSD for safetensors shards: %s", tmp_dir)
  except (FileNotFoundError, PermissionError):
    tmp_dir = tempfile.mkdtemp()
    logger.info("Using default temp directory for safetensors shards: %s", tmp_dir)

  save_sharded_safetensors_by_layer(cpu_state, str(save_dir), tmp_dir=tmp_dir)
  logger.info("Safetensors shards + index written to %s", save_dir)


def maybe_move_to_mounted_gcs(tmp_dir: Path | None, save_dir: str):
  """
  If tmp_dir is provided, move *.safetensors files and index file
  from tmp_dir to save_dir using gsutil or shutil.

  Args:
      tmp_dir (Path): Local directory containing files to upload.
      save_dir (Path): Destination directory (e.g., /tmp/gcs-mount/...)
  """
  if tmp_dir:
    save_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("gsutil"):
      try:
        # gsutil seems to give 8x speedup over shutil.copy2
        logger.info("Using gsutil for upload")
        move_to_mounted_gcs_gsutil(tmp_dir, save_dir)
        return
      except subprocess.CalledProcessError as e:
        logger.warning("gsutil failed: %s. Falling back to shutil-based copy.", str(e))
    else:
      logger.info("gsutil not found. Falling back to shutil-based copy.")
      move_to_mounted_gcs_shutil(tmp_dir, save_dir)
  else:
    logger.warning("No tmp_dir provided, checkpoint already saved in save_dir.")


def move_to_mounted_gcs_gsutil(work_dir: Path, save_dir: str):
  """
  Moves *.safetensors files and index file from work_dir to save_dir,
  using gsutil for efficient upload to a mounted GCS bucket.

  Args:
      work_dir (Path): Local directory containing files to upload.
      save_dir (Path): Destination directory (e.g., /tmp/gcs-mount/...)
  """
  save_dir.mkdir(parents=True, exist_ok=True)
  cmd = [
    "gsutil",
    "-m",  # Enables parallel (multi-threaded) execution for faster copying
    "-q",  # Suppresses all output unless errors occur (quiet mode)
    "cp",  # Copy command
    "-n",  # No-clobber: skip files that already exist at the destination
    *(str(p) for p in work_dir.glob("*.safetensors")),  # All .safetensors files to copy
    str(save_dir) + "/",  # Destination directory in the GCS bucket
  ]
  cmd_idx = [
    "gsutil",
    "-q",  # Quiet mode
    "cp",  # Copy command
    str(work_dir / "model.safetensors.index.json"),  # Source index file
    str(save_dir) + "/",  # Destination directory
  ]
  subprocess.check_call(cmd)
  subprocess.check_call(cmd_idx)
  shutil.rmtree(work_dir, ignore_errors=True)


def move_to_mounted_gcs_shutil(work_dir: Path, save_dir: Path):
  """
  Moves *.safetensors files and index file from work_dir to save_dir,
  where save_dir is a mounted GCS bucket (e.g. via gcsfuse).

  Args:
      work_dir (Path): Local directory containing files to upload.
      save_dir (Path): Destination directory (e.g., /tmp/gcs-mount/...)
  """
  save_dir.mkdir(parents=True, exist_ok=True)

  for file_path in work_dir.glob("*.safetensors"):
    dest = save_dir / file_path.name
    if not dest.exists():  # don't clobber
      shutil.copy2(file_path, dest)

  index_file = work_dir / "model.safetensors.index.json"
  if index_file.exists():
    dest = save_dir / index_file.name
    if not dest.exists():
      shutil.copy2(index_file, dest)

  shutil.rmtree(work_dir, ignore_errors=True)


def copy_hf_config_files(model_path_or_repo: str, save_dir: Path) -> None:
  """Copy configuration files from a Hugging Face model repository or local directory.

  This function downloads specific configuration files from a Hugging Face model
  repository (if `model_path_or_repo` is a repo ID) or copies them from a local
  directory. Only files matching `HF_MODEL_CONFIG_FILES` will be considered.

  Args:
      model_path_or_repo: Either a local path to a model directory or a Hugging Face
          model repo ID (e.g., "meta-llama/Llama-2-7b-hf").
      save_dir: Target directory to save the copied configuration files.
  """
  patterns = HF_MODEL_CONFIG_FILES

  if os.path.isdir(model_path_or_repo):
    model_dir = Path(model_path_or_repo)
  else:
    model_dir = Path(
      huggingface_hub.snapshot_download(
        repo_id=model_path_or_repo, allow_patterns=patterns
      )
    )

  save_dir = Path(save_dir)
  save_dir.mkdir(parents=True, exist_ok=True)

  for name in patterns:
    src = model_dir / name
    if src.exists():
      shutil.copy2(src, save_dir / name)
    else:
      logger.warning(
        "Configuration file %s not found in HF model repo %s", name, model_dir
      )


def save_hf_tokenizer(model_path_or_repo: str, save_dir: Path) -> None:
  """Save a Hugging Face tokenizer to a local directory.

  This function downloads a tokenizer from a Hugging Face model repository or
  loads it from a local directory and saves it to `save_dir`.

  Args:
      model_path_or_repo: Either a local path to a tokenizer directory or a Hugging Face
          model repo ID (e.g., "meta-llama/Llama-2-7b-hf").
      save_dir: Directory where the tokenizer files will be saved.
  """
  tokenizer = AutoTokenizer.from_pretrained(model_path_or_repo)
  save_dir = Path(save_dir)
  save_dir.mkdir(parents=True, exist_ok=True)
  tokenizer.save_pretrained(save_dir)


@contextmanager
def local_path_from_gcs(path_or_repo: str, temp_dir: str | None = None):
  """A context manager to download GCS content to a local temporary directory.

  If the input `path_or_repo` starts with 'gs://', this function will download
  the contents of the GCS directory to a temporary local directory using the
  `gsutil` command-line tool. The local directory will be automatically cleaned
  up when the context is exited.

  If the input is not a GCS path, it is assumed to be a local path or a
  Hugging Face repository ID, and is yielded unmodified with no cleanup.

  Args:
      path_or_repo: The path to resolve. Can be a GCS URI (e.g.,
        'gs://bucket/data') or a local file path.
      temp_dir: An optional path to a directory for creating the temporary
        download location. If None, the system's default temporary directory
        is used.

  Yields:
      A string containing the path to the local directory.
  """
  if not path_or_repo.startswith("gs://"):
    yield path_or_repo
    return

  if not shutil.which("gsutil"):
    raise RuntimeError(
      "gsutil command not found, but is required for downloading from GCS. "
      "Please install the Google Cloud SDK."
    )

  local_dir = tempfile.mkdtemp(dir=temp_dir)
  try:
    gcs_path = path_or_repo.rstrip("/") + "/*"
    command = ["gsutil", "-m", "-q", "cp", "-r", gcs_path, local_dir]
    subprocess.run(command, check=True, capture_output=True, text=True)
    logger.info(
      "Successfully downloaded files from %s to temporary directory %s.",
      path_or_repo,
      local_dir,
    )

    yield local_dir
  except subprocess.CalledProcessError as e:
    logger.error("gsutil download failed for %s. Stderr:\n%s", path_or_repo, e.stderr)
    raise
  finally:
    logger.info(f"Cleaning up temporary directory: {local_dir}")
    shutil.rmtree(local_dir, ignore_errors=True)
