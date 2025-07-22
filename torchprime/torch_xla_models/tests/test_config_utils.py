import pytest
from omegaconf import OmegaConf

from torchprime.torch_xla_models.utils.config_utils import config_vaidator


@pytest.mark.parametrize(
  "ici_mesh, lb_cp_enabled, attention_kernel, errorMsg, context",
  [
    (
      {"data": 1, "fsdp": 1, "tensor": 1, "context": 2},
      True,
      "splash_attention",
      None,
      2,
    ),
    (
      {"data": 1, "fsdp": 1, "tensor": 1, "context": 2},
      False,
      "flash_attention",
      None,
      2,
    ),
    (
      {"data": 1, "fsdp": 1, "tensor": 1, "context": 2},
      True,
      "flash_attention",
      "Load balanced context parallelism is only supported with splash attention kernel",
      2,
    ),
    (
      {"data": 1, "fsdp": 1, "tensor": 1, "context": 2},
      False,
      "flash_attention",
      "ici context size should equal to model context parallelism size",
      1,
    ),
  ],
)
def test_validate_context_parallelism(
  ici_mesh, lb_cp_enabled, attention_kernel, errorMsg, context
):
  config = custom_config_creator(
    ici_mesh=ici_mesh,
    lb_cp_enabled=lb_cp_enabled,
    attention_kernel=attention_kernel,
    context=context,
  )
  if errorMsg is None:
    config_vaidator(config)
  else:
    try:
      config_vaidator(config)
      raise AssertionError("RuntimeError was not raised!")
    except RuntimeError as e:
      assert errorMsg in str(e)
    except Exception:
      raise AssertionError("RuntimeError was not raised!")  # noqa: B904


def custom_config_creator(
  ici_mesh, lb_cp_enabled=False, attention_kernel=None, context=2
):
  return OmegaConf.create(
    {
      "model": {
        "pure_modules": [],
        "remat": {
          "activation_checkpoint_layers": [],
          "optimization_barrier_layers": [],
          "scan_layers": None,
          "offload_tensors": [],
        },
        "sharding": {"type": "spmd"},
        "load_balance_cp": lb_cp_enabled,
        "context": context,
      },
      "data": {"name": "dummy_dataset", "block_size": 4},
      "task": {
        "name": "dummy_task",
        "global_batch_size": 4,
        "max_steps": 2,
        "optimizer": {"type": "adafactor", "learning_rate": 1e-3},
        "max_grad_norm": None,
        "max_grad_value": None,
        "lr_scheduler": {"type": "constant", "warmup_steps": 0},
      },
      "run_name": None,
      "output_dir": "/tmp/test_output",
      "logging_steps": 1,
      "profile_start_step": -1,
      "profile_end_step": -1,
      "profile_dir": "/tmp/profile",
      "ici_mesh": ici_mesh,
      "dcn_mesh": {},
      "attention_kernel": attention_kernel,
    }
  )
