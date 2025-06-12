"""Flax interop."""

from typing import Any

import jax
import jax.numpy as jnp
import torch
import torchax as tx
from flax import linen as nn
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec
from omegaconf import DictConfig
from torchax.flax import FlaxNNModule


def get_dense_embed_module(cfg: DictConfig) -> torch.nn.Module:
  num_embeddings = sum(cfg.num_embeddings_per_feature)
  features_dim = cfg.model.sparse_feature_size
  batch_size = cfg.batch_size

  env = tx.default_env()
  nnx_emb = nn.Embed(num_embeddings=num_embeddings, features=features_dim)
  sample_input = jnp.ones((batch_size, batch_size), dtype=jnp.int32)

  shard_cfg = cfg.model.sharding.embedding

  if shard_cfg:

    def _set_device_mesh_num(val: Any) -> int:
      if isinstance(val, str) and val == "all":
        return jax.device_count()
      return int(val)

    device_mesh_tuple = tuple(
      _set_device_mesh_num(device) for device in shard_cfg.mesh.devices
    )

    mesh = jax.sharding.Mesh(
      mesh_utils.create_device_mesh(device_mesh_tuple),
      axis_names=tuple(shard_cfg.mesh.axis),
    )
    sharding = jax.sharding.NamedSharding(
      mesh, PartitionSpec(*shard_cfg.partition_spec)
    )

    orig_init_fn = nnx_emb.init

    def sharded_init(prng, *sample_args, **sample_kwargs):
      return jax.jit(orig_init_fn, out_shardings=sharding)(
        prng, *sample_args, **sample_kwargs
      )

    nnx_emb.init = sharded_init

  emb_module = FlaxNNModule(env, nnx_emb, (sample_input,)).to("jax")
  return emb_module


def reduce_embeddings_jax(all_batch_embeddings, features_lengths):
  import jax.numpy as jnp

  num_segments = len(features_lengths)
  segment_ids = jnp.repeat(
    jnp.arange(num_segments),
    features_lengths,
    total_repeat_length=all_batch_embeddings.shape[0],
  )
  r = jax.ops.segment_sum(all_batch_embeddings, segment_ids, num_segments=num_segments)
  return r
