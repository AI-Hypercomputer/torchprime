import pytest

import torch
import torchax
import torchax.interop
from torchprime.experimental.torchax_models.deepseek_v3 import model as ds_model


@pytest.mark.deepseek
def test_moe_can_jit():
  torchax.enable_globally()
  torch.manual_seed(42)
  max_seq_len = 512  # 8192
  vocab_size = 128  # 32000
  n_layer = 1
  n_heads = 4
  dim = 8
  block_size = 16  # 2048
  with torch.no_grad():
    x = torch.ones((1, max_seq_len, 2048), dtype=torch.float32, device='jax')
    model_args = ds_model.ModelArgs()
    model = ds_model.MoE(model_args).to('jax')

    jitted = torchax.interop.JittableModule(model)
    print(jitted(x))
  torchax.disable_globally()

      
