import numpy as np
import torch
import torch_xla
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from omegaconf import DictConfig


def reorder_sequence(
  tensor, cp_size: int, seq_dim: int = 1, to_contiguous: bool = False
):
  """Reorders the sequence of the tensor based on context parallelism
  size.
  cp_size: context parallelism size
  seq_dim: the dimension index of sequence length
  to_contiguous: whether to reorder back the sequence
  For example, with cp_size=2, and to_contiguous
  is false [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 6, 7, 2, 3, 4, 5]
  when cp_size = 2 and to_contiguous is True, reorder it back
  [0, 1, 6, 7, 2, 3, 4, 5] -> [0, 1, 2, 3, 4, 5, 6, 7]
  A sequence will be divied into 2*cp_size of chunks and reorder
  it to couple the chunks of largest & smallest computation
  intensity
  """

  device = torch_xla.device()

  if tensor is None:
    return tensor

  seq_len = tensor.shape[seq_dim]
  group_size = seq_len // (2 * cp_size)

  if cp_size % 2 != 0:
    raise ValueError(f"{cp_size=} must be a multiple of 2.")

  # Need to ensure we have 2 pairs to swap for balancing between cp ranks
  if seq_len % (cp_size * 2) != 0:
    raise ValueError(f"{tensor.shape=} is not a multiple of {cp_size*2=}")

  # [B, S, H, D]: [B, 2*cp_size, S/2*cp_size, H, D] -> [B, 2, S/2*cp_size, H, D]
  # [S, B, H, D]: [2*cp_size, S/2*cp_size, B, H, D] -> [2, S/2*cp_size, B, H, D]
  ori_tensor_shape = tensor.shape
  reshaped = tensor.reshape(
    *ori_tensor_shape[:seq_dim],
    2 * cp_size,
    group_size,
    *ori_tensor_shape[seq_dim + 1 :],
  )

  if not to_contiguous:
    # Create first and second halves
    first_half = torch.arange(cp_size).to(device)
    second_half = torch.arange(2 * cp_size - 1, cp_size - 1, -1).to(device)

    # Stack and reshape to interleave
    src_indices = torch.stack([first_half, second_half], axis=1).reshape(-1).to(device)

  else:
    half = cp_size // 2

    # Build the 1st and 2nd groups of contiguous‑pair indices:
    first_pair = [4 * r for r in range(half)]
    second_pair = [4 * r + 2 for r in range(half)]
    third_pair = [2 * cp_size - 1 - 4 * r for r in range(half)]
    fourth_pair = [i - 2 for i in third_pair]

    # Concatenate so each rank’s two indices sit next to each other:
    first_block = first_pair + third_pair
    second_block = second_pair + fourth_pair

    # Stack into shape (2*cp_size//2, 2) → then flatten → length=2*cp_size
    src_indices = (
      torch.stack([torch.tensor(first_block), torch.tensor(second_block)], axis=1)
      .reshape(-1)
      .to(device)
    )

  # One gather and one reshape
  reordered = torch.index_select(input=reshaped, dim=seq_dim, index=src_indices).to(
    device
  )

  # Reshape back to original dimensions
  return reordered.reshape(ori_tensor_shape)


class LoadBalancedCausalMask(splash_attention_mask._ComputableMask):
  """Lazy causal mask, prevents the model from attending to future tokens.
  Attributes:
    offset: Offset of q start wrt kv. A positive offset shifts the bottom
      triangle upward, a negative one shifts it downward. A negative offset
      makes the first 'offset' rows of the attention matrix all 0s which leads
      to undefined softmax.
  """

  offset: int
  shape: tuple[int, int]
  cp_size: int

  def __init__(
    self,
    shape: tuple[int, int],
    offset: int = 0,
    shard_count: int = 1,
    cp_size: int = 4,
  ):
    self.offset = offset

    def causal_mask_function(q_ids, kv_ids):
      if self.offset == 0:
        return q_ids >= kv_ids
      else:
        return q_ids + self.offset >= kv_ids

    arr = np.arange(shape[0])
    # we reorder the mask to be load balanced following the same approach as
    # used to reorder the input tokens
    out = reorder_mask(
      tensor=arr[np.newaxis, :, np.newaxis, np.newaxis],
      cp_size=cp_size,
      seq_dim=1,
    )
    q_sequence = out[0, :, 0, 0]

    mask_function = causal_mask_function

    super().__init__(
      shape=shape,
      mask_function=mask_function,
      shard_count=shard_count,
    )
    self.q_sequence = q_sequence

  def __eq__(self, other: object):
    if not isinstance(other, type(self)):
      return NotImplemented

    return (
      self.shape == other.shape
      and self.offset == other.offset
      and np.array_equal(self.q_sequence, other.q_sequence)
    )

  def __hash__(self):
    return hash(
      (
        type(self),
        self.shape,
        self.offset,
        self.q_sequence.tobytes() if self.q_sequence is not None else None,
      )
    )


def cp_enabled(config: DictConfig):
  """
  Check if context parallelism is enabled
  """
  return "context" in config.ici_mesh and config.ici_mesh.context > 1


def lb_cp_enabled(config: DictConfig):
  """
  Check if load balanced context parallelism is enabled
  """
  return (
    cp_enabled(config)
    and "load_balance_cp" in config.model
    and config.model.load_balance_cp
  )


def reorder_mask(tensor, cp_size: int, seq_dim: int):
  seq_len = tensor.shape[seq_dim]
  group_size = seq_len // (2 * cp_size)

  if cp_size % 2 != 0:
    raise ValueError(f"{cp_size=} must be a multiple of 2.")

  # Need to ensure we have 2 pairs to swap for balancing between cp ranks
  if seq_len % (cp_size * 2) != 0:
    raise ValueError(f"{tensor.shape=} is not a multiple of {cp_size*2=}")

  # [B, S, H, D]: [B, 2*cp_size, S/2*cp_size, H, D] -> [B, 2, S/2*cp_size, H, D]
  # [S, B, H, D]: [2*cp_size, S/2*cp_size, B, H, D] -> [2, S/2*cp_size, B, H, D]
  ori_tensor_shape = tensor.shape
  reshaped = tensor.reshape(
    *ori_tensor_shape[:seq_dim],
    2 * cp_size,
    group_size,
    *ori_tensor_shape[seq_dim + 1 :],
  )

  # Create first and second halves
  first_half = np.arange(cp_size)
  second_half = np.arange(2 * cp_size - 1, cp_size - 1, -1)

  # Stack and reshape to interleave
  src_indices = np.stack([first_half, second_half], axis=1).reshape(-1)

  # One gather and one reshape
  reordered = np.take(reshaped, src_indices, axis=seq_dim)

  # Reshape back to original dimensions
  return reordered.reshape(ori_tensor_shape)
