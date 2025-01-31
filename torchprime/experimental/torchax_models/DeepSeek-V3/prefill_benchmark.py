import functools
import time

import jax
import torch
import torchax
import torchax.interop
from model import ModelArgs, Transformer
from torchax.interop import JittableModule


def single_device_compile():
  print("======= single_device_compile =======")
  torch.set_default_dtype(torch.bfloat16)
  env = torchax.default_env()
  torch.manual_seed(42)
  torchax.enable_performance_mode()

  args = ModelArgs()

  with torch.no_grad(), env:
    x = torch.randint(0, args.vocab_size, (1, 2048))
    x = x.to("jax")
    model = Transformer(args)
    model.to("jax")
    model.embed = JittableModule(model.embed)
    # for i in range(len(model.layers)):
    #     model.layers[i] = JittableModule(model.layers[i])
    model.norm = JittableModule(model.norm)
    model.head = JittableModule(model.head)

    for i in range(5):
      step_start = time.perf_counter()
      logits = model(x, 0)
      jax.block_until_ready(torchax.tensor.t2j(logits))
      step_end = time.perf_counter()
      print(
        i,
        "step latency: ",
        step_end - step_start,
      )


def single_device_eager():
  print("======= single_device_eager =======")
  torch.set_default_dtype(torch.bfloat16)
  env = torchax.default_env()
  torch.manual_seed(42)
  torchax.enable_performance_mode()

  args = ModelArgs()

  with torch.no_grad(), env:
    x = torch.randint(0, args.vocab_size, (1, 2048))
    x = x.to("jax")
    model = Transformer(args)
    model.to("jax")
    weights = model.state_dict()
    model_forward = functools.partial(torch.func.functional_call, model)
    # model_forward = torchax.interop.jax_jit(model_forward)

    for i in range(5):
      step_start = time.perf_counter()
      logits = model_forward(weights, (x, 0))
      jax.block_until_ready(torchax.tensor.t2j(logits))
      step_end = time.perf_counter()
      print(
        i,
        "step latency: ",
        step_end - step_start,
      )


def main(option="single_device_eager"):
  if option == "single_device_eager":
    single_device_eager()
  elif option == "single_device_compile":
    single_device_compile()
  else:
    raise Exception("Invalid option")


if __name__ == "__main__":
  import fire

  fire.Fire(main)
