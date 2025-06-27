from omegaconf import OmegaConf

from torchprime.torch_xla_models.utils.config_utils import config_vaidator


def test_validate_context_parallelism():
  # test correct config
  ici_mesh = ({"data": 1, "fsdp": 1, "tensor": 1, "context": 2},)
  config = custom_config_creator(
    ici_mesh=ici_mesh, lb_cp_enabled=True, attention_kernel="splash_attention"
  )
  config_vaidator(config)

  # test correct config when lb_cp is disabled
  ici_mesh = ({"data": 1, "fsdp": 1, "tensor": 1, "context": 2},)
  config = custom_config_creator(
    ici_mesh=ici_mesh, lb_cp_enabled=False, attention_kernel="flash_attention"
  )
  config_vaidator(config)

  # test incorrect config when wrong kernel is used
  ici_mesh = ({"data": 1, "fsdp": 1, "tensor": 1, "context": 2},)
  config = custom_config_creator(
    ici_mesh=ici_mesh, lb_cp_enabled=True, attention_kernel="flash_attention"
  )
  try:
    config_vaidator(config)
    raise AssertionError("RuntimeError was not raised!")
  except RuntimeError as e:
    assert (
      "Load balanced context parallelism is only supported with splash attention kernel"
      in str(e)
    )
  except Exception:
    raise AssertionError("RuntimeError was not raised!")  # noqa: B904


def custom_config_creator(ici_mesh, lb_cp_enabled=False, attention_kernel=None):
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
      "load_balance_cp": lb_cp_enabled,
      "attention_kernel": attention_kernel,
    }
  )
