from torchprime.utils.parallelism_utils import lb_cp_enabled


def config_vaidator(config: dict):
  """
  This validator checks whether the user provided config is valid
  in advance, thus avoiding unnecessary unclear failure or misuses,
  improving usability.
  """
  if lb_cp_enabled(config) and config.attention_kernel != "splash_attention":
    raise RuntimeError(
      "Load balanced context parallelism is only supported with splash attention kernel"
    )
