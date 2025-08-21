import copy
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch_xla
import transformers.models.llama4.modeling_llama4 as hf_modeling_llama4
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig
from transformers import Llama4ForCausalLM as HfLlama4ForCausalLM

import torchprime.torch_xla_models.model.llama4.model as modeling_llama4
from torchprime.torch_xla_models.model.llama4 import Llama4TextForCausalLM
from torchprime.torch_xla_models.tests.test_utils import (
  get_forward_and_backward_outputs,
)


@dataclass
class LlamaFixture:
  vocab_size: int
  hf_model: HfLlama4ForCausalLM
  model: Llama4TextForCausalLM
  model_config: DictConfig


def get_llama_4_text_dummy_model(
  moe_implementation: modeling_llama4.MoeImplementation = None,
) -> LlamaFixture:
  torch.manual_seed(42)
  torch_xla.manual_seed(42)
  vocab_size = 128
  config_path = Path(__file__).parent / "hf_model_config" / "llama-4-scout-17b-16e"
  config = AutoConfig.from_pretrained(config_path)
  config.text_config.num_hidden_layers = 1
  config.text_config.num_local_experts = 4
  config.text_config.moe_layers = [0]
  config.text_config.vocab_size = vocab_size
  config.text_config.head_dim = 64
  config.text_config.num_attention_heads = 8
  config.text_config.hidden_size = 512
  config.text_config.intermediate_size = 64
  config.text_config.pad_token_id = 127
  config.text_config.use_cache = False
  config.flash_attention = False

  torchprime_config = OmegaConf.create(config.to_dict())
  torchprime_config.vision_config = None
  del torchprime_config.text_config.rope_scaling.original_max_position_embeddings
  del torchprime_config.text_config.rope_scaling.rope_type
  torchprime_config.text_config.attention_kernel = None
  torchprime_config.text_config.moe_implementation = moe_implementation

  # Place model on CPU device first
  with torch.device("cpu"):
    hf_text_model = HfLlama4ForCausalLM(config.text_config)
    hf_text_model.init_weights()
    text_model = Llama4TextForCausalLM(torchprime_config.text_config)
    text_model.load_state_dict(hf_text_model.state_dict())
  return LlamaFixture(vocab_size, hf_text_model, text_model, torchprime_config)


def noop(mod):
  return mod


def scan_decoders(mod):
  import torchprime.torch_xla_models.scan_layers

  return torchprime.torch_xla_models.scan_layers.compile(mod, "model.layers")


@pytest.mark.parametrize(
  "fixture",
  [get_llama_4_text_dummy_model],
  ids=["Llama4 text dummy"],
)
@pytest.mark.parametrize("transform", [noop, scan_decoders])
@pytest.mark.parametrize("input_size", [8, 128, 256])
def test_forward_and_backward_our_model_against_hf_model(
  fixture, transform, input_size
):
  """Compares the numerical consistency of our Llama4 model against the
  Hugging Face reference on an XLA device.

  Asserts that logits, loss, and gradients are nearly identical after a
  full forward and backward pass.
  """
  # Arrange
  fixture = fixture()
  device = torch.device("xla")
  model_xla = copy.deepcopy(fixture.model).to(device)
  model_xla = transform(model_xla)
  hf_model_xla = copy.deepcopy(fixture.hf_model).to(device)
  torch_xla.sync()
  input_ids = torch.randint(fixture.vocab_size, ((2, input_size // 2))).to(device)
  attention_mask = torch.ones_like(input_ids)

  # Act
  (hf_logits, hf_loss), hf_params = get_forward_and_backward_outputs(
    hf_model_xla,
    input_ids=input_ids,
    labels=input_ids,
    attention_mask=attention_mask,
  )
  (model_logits, model_loss), model_params = get_forward_and_backward_outputs(
    model_xla,
    input_ids=input_ids,
    labels=input_ids,
    attention_mask=attention_mask,
  )

  # Assert
  torch.testing.assert_close(
    hf_logits, model_logits, atol=1e-4, rtol=1e-9, msg="Logits are not equal"
  )
  torch.testing.assert_close(
    hf_loss, model_loss, atol=1e-6, rtol=1e-9, msg="Losses are not equal"
  )
  for (name_hf, p_hf), (name_model, p_model) in zip(
    hf_params, model_params, strict=True
  ):
    assert name_hf == name_model, f"Parameter name mismatch: {name_hf} vs {name_model}"
    assert p_hf.grad is not None, f"Gradient for {name_hf} is None in hf_model"
    assert p_model.grad is not None, f"Gradient for {name_model} is None in model"
    torch.testing.assert_close(
      p_hf.grad,
      p_model.grad,
      atol=1e-6,
      rtol=1e-9,
      msg=f"Gradients for '{name_hf}' differ",
    )


@pytest.mark.parametrize(
  "fixture",
  [get_llama_4_text_dummy_model],
  ids=["Llama4 text dummy"],
)
def test_forward_and_backward_torch_xla_against_native(fixture):
  """Compares the numerical consistency of our Llama4 model on native CPU
  vs. an XLA device.

  Asserts that logits, loss, and gradients are nearly identical after a
  full forward and backward pass on both backends.
  """
  # Arrange
  fixture = fixture()
  input_size = 8
  cpu_device = torch.device("cpu")
  input_ids = torch.randint(
    fixture.vocab_size, ((2, input_size // 2)), device=cpu_device
  )
  attention_mask = torch.ones_like(input_ids)

  # Act
  (logits_native, loss_native), params_native = get_forward_and_backward_outputs(
    fixture.model,
    input_ids=input_ids,
    labels=input_ids,
    attention_mask=attention_mask,
  )
  (logits_xla, loss_xla), params_xla = get_forward_and_backward_outputs(
    copy.deepcopy(fixture.model).to(torch_xla.device()),
    input_ids=input_ids.to(torch_xla.device()),
    labels=input_ids.to(torch_xla.device()),
    attention_mask=attention_mask.to(torch_xla.device()),
  )

  # Assert
  torch.testing.assert_close(
    logits_native,
    logits_xla.to("cpu"),
    atol=1e-2,
    rtol=1e-6,
    msg="CPU run and XLA run logits are not equal",
  )
  torch.testing.assert_close(
    loss_native,
    loss_xla.to("cpu"),
    atol=1e-2,
    rtol=1e-6,
    msg="CPU run and XLA run loss is not equal",
  )
  for (name_native, p_native), (name_xla, p_xla) in zip(
    params_native, params_xla, strict=True
  ):
    assert name_native == name_xla
    assert p_native.grad is not None
    assert p_xla.grad is not None
    torch.testing.assert_close(
      p_native.grad,
      p_xla.grad.cpu(),
      atol=1e-2,
      rtol=1e-4,
      msg=f"Gradients for '{name_native}' differ between Native and XLA",
    )


@pytest.mark.parametrize(
  "fixture",
  [get_llama_4_text_dummy_model],
  ids=["Llama4 text dummy"],
)
def test_rotarty_embedding_our_model_against_hf_model_native(fixture):
  fixture = fixture()
  device = torch.device("xla")
  model_xla = copy.deepcopy(fixture.model).to(device)
  torch_xla.sync()
  hf_model = fixture.hf_model
  input_sizes = [8, 128, 256]
  for seq_len in input_sizes:
    batch_size = 2
    in_embedding = torch.randn(
      (batch_size, seq_len // batch_size, fixture.model.config.hidden_size)
    )
    position_ids = torch.arange(seq_len).unsqueeze(0)
    hf_position_embedings = hf_model.model.rotary_emb(in_embedding, position_ids)
    num_attention_heads = fixture.model_config.text_config.num_attention_heads
    head_dim = fixture.model_config.text_config.head_dim
    xq = torch.rand(batch_size, seq_len, num_attention_heads, head_dim)
    xk = torch.rand(batch_size, seq_len, num_attention_heads, head_dim)

    hf_xq_out, hf_xk_out = hf_modeling_llama4.apply_rotary_emb(
      xq, xk, hf_position_embedings
    )

    position_embedings = model_xla.model.rotary_emb(
      in_embedding.to(device), position_ids.to(device)
    )
    xq_out, xk_out = modeling_llama4.apply_rotary_emb(
      xq.to(device), xk.to(device), position_embedings
    )
    torch_xla.sync()
    torch.testing.assert_close(
      hf_xq_out,
      xq_out.to("cpu"),
      atol=1e-3,
      rtol=1e-6,
      msg="xq after apply_rotary_emb is not equal",
    )
    torch.testing.assert_close(
      hf_xk_out,
      xk_out.to("cpu"),
      atol=1e-3,
      rtol=1e-6,
      msg="xk after apply_rotary_emb is not equal",
    )


@pytest.mark.parametrize(
  "fixture",
  [get_llama_4_text_dummy_model],
  ids=["Llama4 dummy model"],
)
def test_moe_layer_from_model(fixture):
  """
  Tests the Llama4TextMoe layer from a loaded model to verify that the
  memory-efficient `forward` method is mathematically equivalent to the
  `forward_original` method.
  """
  # Arrange
  device = torch.device("xla")
  torch.manual_seed(42)

  # Create dummy input
  batch_size = 2
  seq_len = 16

  # Act
  # Model with original MOE layer implementation
  model_xla = copy.deepcopy(fixture().model).to(device)
  model_xla.eval()
  moe_layer = model_xla.model.layers[0].feed_forward
  assert isinstance(moe_layer, modeling_llama4.Llama4TextMoe)
  hidden_states = torch.rand(batch_size, seq_len, model_xla.config.hidden_size).to(
    device
  )
  out_original, scores_original = moe_layer.forward(hidden_states)
  torch_xla.sync()

  # Model with sequential MOE layer implementation
  fixture_seq = fixture(moe_implementation=modeling_llama4.MoeImplementation.sequential)
  model_seq = copy.deepcopy(fixture_seq.model).to(device)
  model_seq.eval()
  moe_layer_seq = model_seq.model.layers[0].feed_forward
  assert isinstance(moe_layer_seq, modeling_llama4.Llama4TextMoeSequential)
  out_seq, scores_seq = moe_layer.forward(hidden_states)
  torch_xla.sync()

  # Model with Expert Parallelism implementation
  fixture_exp = fixture(
    moe_implementation=modeling_llama4.MoeImplementation.expert_parallelism
  )
  model_exp = copy.deepcopy(fixture_exp.model).to(device)
  model_exp.eval()
  moe_layer_exp = model_exp.model.layers[0].feed_forward
  assert isinstance(moe_layer_exp, modeling_llama4.Llama4TextMoeExpertParallelism)
  out_exp, scores_exp = moe_layer.forward(hidden_states)
  torch_xla.sync()

  # Assert
  assert out_original.size() == out_seq.size(), (
    "Dimensions for outputs tensors don't match for seq MOE"
  )
  assert scores_original.size() == scores_seq.size(), (
    "Dimensions for scores don't match for seq MOE"
  )

  torch.testing.assert_close(
    out_original,
    out_seq,
    atol=1e-5,
    rtol=1e-6,
    msg="sequential MoE layer output does not match original logic",
  )
  torch.testing.assert_close(
    scores_original,
    scores_seq,
    atol=1e-5,
    rtol=1e-6,
    msg="sequential MoE layer scores do not match original logic",
  )

  assert out_original.size() == out_exp.size(), (
    "Dimensions for outputs tensors don't match for expert parallelism MOE"
  )
  assert scores_original.size() == scores_exp.size(), (
    "Dimensions for scores don't match for expert parallelism MOE"
  )

  torch.testing.assert_close(
    out_original,
    out_exp,
    atol=1e-5,
    rtol=1e-6,
    msg="expert parallelism MoE layer output does not match original logic",
  )
  torch.testing.assert_close(
    scores_original,
    scores_exp,
    atol=1e-5,
    rtol=1e-6,
    msg="expert parallelism MoE layer scores do not match original logic",
  )
