from torchprime.utils.parallelism_utils import cp_enabled, lb_cp_enabled


def config_vaidator(config: dict):
  """
  This validator checks whether the user provided config is valid
  in advance, thus avoiding unnecessary unclear failure or misuses,
  improving usability.
  """
  fsdp_size, dp_size = 1, 1
  if hasattr(config, "ici_mesh"):
    fsdp_size *= getattr(config.ici_mesh, "fsdp", 1)
    dp_size *= getattr(config.ici_mesh, "data", 1)
  if hasattr(config, "dcn_mesh"):
    fsdp_size *= getattr(config.dcn_mesh, "fsdp", 1)
    dp_size *= getattr(config.dcn_mesh, "data", 1)

  assert config.task.global_batch_size >= fsdp_size * dp_size, (
    f"Global_batch_size ({config.task.global_batch_size}) should be no less than ici/dcn_mesh.fsdp ({fsdp_size}) * ici/dcn_mesh.data ({dp_size})"
  )

  if (
    "load_balance_cp" in config.model
    and config.model.load_balance_cp
    and not cp_enabled
  ):
    raise RuntimeError(
      "Load balanced context parallelism can only be used when cp is enabled"
    )

  if lb_cp_enabled(config) and config.attention_kernel != "splash_attention":
    raise RuntimeError(
      "Load balanced context parallelism is only supported with splash attention kernel"
    )

  if cp_enabled(config):
    if "context" not in config.model:
      raise RuntimeError("Specify context parallelism size in config.model as well")
    elif config.model.context != config.ici_mesh.context:
      raise RuntimeError(
        "ici context size should equal to model context parallelism size"
      )
