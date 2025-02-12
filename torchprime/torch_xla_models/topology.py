"""
`topology` obtains information on the accelerator topology in the current environment.
"""

from functools import lru_cache

import torch_xla.runtime as xr


@lru_cache
def is_multi_slice() -> bool:
  """
  Check if we are running in multi-slice/pod environment.
  """
  num_slices = get_num_slices()
  return num_slices > 1


@lru_cache
def get_num_slices():
  """
  Get the number of slices (ICI connected groups of TPUs) in the environment.
  """
  device_attributes = xr.global_runtime_device_attributes()
  slice_indices = []
  for d in device_attributes:
    index = d.get("slice_index", 0)
    assert isinstance(
      index, int
    ), f"Expected slice_index to be an int, got {type(index)} {index}"
    slice_indices.append(index)
  num_slices = max(slice_indices) + 1
  return num_slices


def is_1d_sharding(mesh_values: tuple[int, ...]) -> bool:
  """
  Check if the devices are distributed along exactly one dimension of the virtual device mesh.
  """
  non_trivial = [v for v in mesh_values if v > 1]
  if len(non_trivial) == 0:
    # Special case if the mesh looks like `[1, 1, 1, 1]`.
    return True
  return len(non_trivial) == 1
