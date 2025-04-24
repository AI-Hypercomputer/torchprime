import time

import torch
from torch.utils import _pytree as pytree

from torchprime import models


def run_model_torchax(model, batch_size, number_of_runs, eager):
  import jax
  import torchax as tx
  import torchax.interop
  torchax.enable_globally()
  model = model.to('jax')

  wait_val = tx.interop.torch_view(jax.block_until_ready)

  if not eager:
    model = tx.interop.JittableModule(model)


  for i in range(number_of_runs):
    print('Iteration ', i)
    inputs = model.get_sample_inputs(batch_size)
    args, kwargs = pytree.tree_map_only(torch.Tensor, lambda t: t.to('jax'), inputs)
    args, kwargs = wait_val((args, kwargs))
    start = time.perf_counter()
    res = model.forward(*args, **kwargs)
    wait_val(res)
    print(res)
    end = time.perf_counter()
    print(f'Iteration {i} took: {end - start}s')


def run_model_xla(model, batch_size, number_of_runs):
  import torch_xla
  model = model.to('xla')

  for i in range(number_of_runs):
    print('Iteration ', i)
    inputs = model.get_sample_inputs(batch_size)
    args, kwargs = pytree.tree_map_only(torch.Tensor, lambda t: t.to('xla'), inputs)
    torch_xla.sync(wait=True)
    start = time.perf_counter()
    res = model.forward(*args, **kwargs)
    torch_xla.sync(wait=True)

    print(res)
    end = time.perf_counter()
    print(f'Iteration {i} took: {end - start}s')


def main(model_id, run_type='torch_xla', batch_size=2, number_of_runs=5):
  print('Running with flags:')
  for k, v in locals().items():
    print(f'{k}: {v}')

  model_factory = models.registry.get(model_id)
  assert model_factory is not None, 'Model with id {model_id} not registered'

  model = model_factory()
  print('Model init successful')

  if run_type == 'torch_xla':
    run_model_xla(model, batch_size, number_of_runs)
  elif run_type == 'torchax_eager':
    run_model_torchax(model, batch_size, number_of_runs, eager=True)
  elif run_type == 'torchax':
    run_model_torchax(model, batch_size, number_of_runs, eager=False)
  else:
    raise AssertionError(f'run_type: {run_type} unknown. Please pass in torch_xla, torchax or torchax_eager')

  print('Model run successful')

    
if __name__ == '__main__':
  import fire
  fire.Fire(main)