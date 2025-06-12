import jax
import torch
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec
from torchax import interop

all_data_parallel_sharding = None


def get_dataparallel_sharding():
  global all_data_parallel_sharding
  if all_data_parallel_sharding is None:
    num_of_partitions = jax.device_count()
    mesh = jax.sharding.Mesh(
      mesh_utils.create_device_mesh((num_of_partitions,)),
      axis_names=("data",),
    )
    all_data_parallel_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec("data"))
  return all_data_parallel_sharding


def make_data_parallel(x: torch.Tensor):
  return interop.call_jax(
    jax.lax.with_sharding_constraint,
    x,
    get_dataparallel_sharding(),
  )
