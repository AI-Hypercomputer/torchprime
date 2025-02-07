import pytest


@pytest.mark.deepseek
def test_moe_can_jit():
  import torch
  import torchax
  import torchax.interop

  from torchprime.experimental.torchax_models.deepseek_v3 import model as ds_model

  torchax.enable_globally()
  torch.manual_seed(42)
  max_seq_len = 512  # 8192
  with torch.no_grad():
    x = torch.ones((1, max_seq_len, 2048), dtype=torch.float32, device="jax")
    model_args = ds_model.ModelArgs()
    model = ds_model.MoE(model_args).to("jax")

    jitted = torchax.interop.JittableModule(model)
    print(jitted(x))
  torchax.disable_globally()
