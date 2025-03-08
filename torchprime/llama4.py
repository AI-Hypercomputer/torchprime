import torch
from torchprime.experimental.torchax_models.llama_new import core_transformer
from torchprime.experimental.torchax_models.llama_new import args
from torchprime.experimental.torchax_models.llama_new import datatypes
import torchax
import torchax.interop

torchax.enable_globally()

env = torchax.default_env()
env.config.debug_print_each_op = True


model_args = args.make_17b(max_batch_size=1, max_seq_len=2048, tiny=True)

model = core_transformer.CoreTransformer(model_args)
model.to('jax')

tokens = torch.arange(0, 10, device='jax').reshape((1, 10))

@torchax.interop.jax_jit
def eval_model(weights, tokens):
  inputs = datatypes.CoreTransformerInput(tokens, 0)
  res = torch.func.functional_call(model, weights, (inputs, ))
  return res.logits

  
import jax
def f():
  return torch.arange(0, 1000, device='jax').jax()

print(jax.jit(f).lower().as_text('hlo'))

weights = model.state_dict()

print(eval_model(weights, tokens))



print(model)



