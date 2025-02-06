import copy
import functools
import unittest

import pytest
import torch
import torch_xla
from omegaconf import OmegaConf
from transformers import AutoConfig
from transformers import LlamaForCausalLM as HfLlamaForCausalLM

from torchprime.torch_xla_models.llama import LlamaForCausalLM
from torchprime.torch_xla_models.llama.model import LlamaDecoderLayer


class TestYourModule(unittest.TestCase):
  def setUp(self):
    super().setUp()
    torch.manual_seed(42)
    torch_xla.manual_seed(42)
    self.vocab_size = 128
    config = AutoConfig.from_pretrained(
      "meta-llama/Meta-Llama-3-8B",
      num_hidden_layers=1,
      num_attention_heads=8,
      hidden_size=8,
      intermediate_size=16,
      vocab_size=self.vocab_size,
    )
    config.flash_attention = False
    torchprime_config = OmegaConf.create(
      {
        "vocab_size": 128,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "hidden_act": "silu",
        "max_position_embeddings": 8192,
        "initializer_range": 0.02,
        "rms_norm_eps": 1.0e-05,
        "attention_dropout": False,
        "attention_bias": False,
        "flash_attention": False,
        "rope_theta": 500000.0,
      }
    )
    # place model on CPU device first
    with torch.device("cpu"):
      self.hf_model = HfLlamaForCausalLM(config)
      self.model = LlamaForCausalLM(torchprime_config)
      self.model.load_state_dict(self.hf_model.state_dict())

  def test_forward_our_model_against_hf_model(self):
    device = torch_xla.device()
    model_xla = copy.deepcopy(self.model).to(device)
    hf_model_xla = copy.deepcopy(self.hf_model).to(device)
    torch_xla.sync()
    input_sizes = [8, 128, 256]
    for input_size in input_sizes:
      input = torch.randint(128, ((2, input_size // 2))).to(device)
      hf_output = hf_model_xla(
        input, labels=input, attention_mask=torch.ones_like(input)
      )
      llama_xla_logits, llama_xla_loss = model_xla(
        input, labels=input, attention_mask=torch.ones_like(input)
      )
      torch_xla.sync()
      self.assertTrue(
        torch.allclose(hf_output.logits, llama_xla_logits, atol=1e-6),
        "logits are not equal",
      )
      self.assertTrue(
        torch.allclose(hf_output.loss, llama_xla_loss, atol=1e-6),
        "loss is not equal",
      )

  def test_forward_torch_xla_against_native(self):
    input_size = 8
    device = torch.device("cpu")
    input = torch.randint(self.vocab_size, ((2, input_size // 2)))
    llama_native_logits, llama_native_loss = self.model(
      input, labels=input, attention_mask=torch.ones_like(input)
    )

    device = torch_xla.device()
    input = input.to(device)
    model_xla = copy.deepcopy(self.model).to(device)
    torch_xla.sync()

    llama_xla_logits, llama_xla_loss = model_xla(
      input, labels=input, attention_mask=torch.ones_like(input)
    )
    torch_xla.sync()
    self.assertTrue(
      torch.allclose(llama_native_logits, llama_xla_logits.to("cpu"), atol=1e-2),
      "CPU run and XLA run logits are not equal",
    )
    self.assertTrue(
      torch.allclose(llama_native_loss, llama_xla_loss.to("cpu"), atol=1e-2),
      "CPU run and XLA run loss is not equal",
    )


class TestLlamaSpmd(unittest.TestCase):
  def setUp(self):
    import torch_xla.runtime as xr

    xr.use_spmd()
    super().setUp()

  def test_config_sharding_against_fsdp_v2(self):
    """
    Test that the config based sharder has identical behavior to FSDPv2.

    Specifically:
    - Model weights have the same sharding spec
    - Outputs have the same sharding spec and are numerically close
    - Gradients have the same sharding spec and are numerically close
    """
    import numpy as np
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from torch_xla.distributed.spmd import Mesh

    # TODO(https://github.com/pytorch/xla/issues/8063): `xla_force_host_platform_device_count` doesn't
    # work on PyTorch/XLA. We must run this on the TPU for now.
    if xr.device_type() != "TPU":
      pytest.skip("This test only works on TPU")

    super().setUp()
    torch.manual_seed(42)
    torch_xla.manual_seed(42)
    self.vocab_size = 128256
    config = AutoConfig.from_pretrained(
      "meta-llama/Meta-Llama-3-8B",
      num_hidden_layers=2,
      num_attention_heads=32,
      hidden_size=4096,
      intermediate_size=14336,
      vocab_size=self.vocab_size,
    )
    config.flash_attention = False
    torchprime_config = OmegaConf.create(
      {
        "vocab_size": 128256,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_act": "silu",
        "max_position_embeddings": 131072,
        "initializer_range": 0.02,
        "rms_norm_eps": 1.0e-05,
        "attention_dropout": False,
        "attention_bias": False,
        "flash_attention": True,
        "rope_theta": 500000.0,
      }
    )
    # Place model on CPU device first
    with torch.device("cpu"):
      self.hf_model = HfLlamaForCausalLM(config)
      self.model = LlamaForCausalLM(torchprime_config)
      self.model.load_state_dict(self.hf_model.state_dict())

    # Define mesh for test
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices, 1, 1)
    assert num_devices > 1, "The TPU VM should have more than 1 device for SPMD testing"
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("dcn", "fsdp", "tensor", "expert"))
    xs.set_global_mesh(mesh)

    # Create random input of batch size 8.
    input = torch.randint(self.vocab_size, ((8, 128)), device=torch_xla.device())
    xs.mark_sharding(input, mesh, ("fsdp", None))
    torch_xla.sync()

    # Shard our model with config based sharding
    sharding_config = {
      # Weights
      "model.embed_tokens.weight": ["fsdp", None],
      "model.layers.*.self_attn.q_proj.weight": ["fsdp", None],
      "model.layers.*.self_attn.k_proj.weight": [None, "fsdp"],
      "model.layers.*.self_attn.v_proj.weight": [None, "fsdp"],
      "model.layers.*.self_attn.o_proj.weight": ["fsdp", None],
      "model.layers.*.mlp.gate_proj.weight": ["fsdp", None],
      "model.layers.*.mlp.up_proj.weight": ["fsdp", None],
      "model.layers.*.mlp.down_proj.weight": [None, "fsdp"],
      "model.layers.*.input_layernorm.weight": ["fsdp"],
      "model.layers.*.post_attention_layernorm.weight": ["fsdp"],
      "model.norm.weight": ["fsdp"],
      "lm_head.weight": ["fsdp", None],
      # Activations
      "model.layers.*": ["fsdp", None, None],
      "lm_head": ["fsdp", None, None],
    }
    from torchprime.sharding.shard_model import shard_torch_xla_model_from_config

    model_config_sharded = shard_torch_xla_model_from_config(
      copy.deepcopy(self.model).to("xla"), config=sharding_config
    )
    torch_xla.sync()

    # Run the model and backwards
    config_logits, config_loss = model_config_sharded(
      input, labels=input, attention_mask=torch.ones_like(input)
    )
    config_loss.backward()
    torch_xla.sync()

    # Shard model with FSDPv2
    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
      SpmdFullyShardedDataParallel as FSDPv2,
    )

    def shard_output(output, mesh):
      real_output = None
      if isinstance(output, torch.Tensor):
        real_output = output
      elif isinstance(output, tuple):
        real_output = output[0]
      else:
        raise RuntimeError("Unsupported")
      xs.mark_sharding(real_output, mesh, ("fsdp", None, None))

    auto_wrap_policy = functools.partial(
      transformer_auto_wrap_policy,
      # Transformer layer class to wrap
      transformer_layer_cls={LlamaDecoderLayer},
    )
    model_fsdp_v2_sharded = FSDPv2(
      copy.deepcopy(self.model),
      shard_output=shard_output,
      auto_wrap_policy=auto_wrap_policy,
    )

    # Run the model and backwards
    model_fsdp_v2_sharded = model_fsdp_v2_sharded.to("xla")
    fsdp_logits, fsdp_loss = model_fsdp_v2_sharded(
      input, labels=input, attention_mask=torch.ones_like(input)
    )
    fsdp_loss.backward()
    torch_xla.sync()

    # Check sharding and numeric accuracy.
    # Check that the outputs are numerically close.
    torch.testing.assert_close(
      config_logits.cpu(),
      fsdp_logits.cpu(),
      msg="Config sharded and FSDP v2 sharded logits are not equal",
    )
    config_sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(config_logits)
    fsdp_v2_sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(fsdp_logits)
    assert config_sharding_spec == fsdp_v2_sharding_spec
    torch.testing.assert_close(
      config_loss.cpu(),
      fsdp_loss.cpu(),
      msg="Config sharded and FSDP v2 sharded loss are not equal",
    )
    config_sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(config_loss)
    fsdp_v2_sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(fsdp_loss)
    assert config_sharding_spec == fsdp_v2_sharding_spec

    # Check that the weights are sharded the same and gradients are numerically close.
    for (p1_name, p1), (p2_name, p2) in zip(
      model_config_sharded.named_parameters(),
      model_fsdp_v2_sharded.named_parameters(),
      strict=True,
    ):
      assert p1_name.split(".")[-1] == p2_name.split(".")[-1]

      config_sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(p1)
      fsdp_v2_sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(p2)
      assert config_sharding_spec == fsdp_v2_sharding_spec

      assert p1.grad is not None
      assert p2.grad is not None
      torch.testing.assert_close(
        p1.grad.cpu(),
        p2.grad.cpu(),
        msg="Config sharded and FSDP v2 sharded gradients are not equal",
      )

      config_sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(p1.grad)
      fsdp_v2_sharding_spec = torch_xla._XLAC._get_xla_sharding_spec(p2.grad)
      assert config_sharding_spec == fsdp_v2_sharding_spec


if __name__ == "__main__":
  unittest.main()
