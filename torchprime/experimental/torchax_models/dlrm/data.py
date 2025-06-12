# partial re-use of code from
# https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/data/dlrm_dataloader.py

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset


@dataclass
class SparseFeatures:
  """
  similuating KayedJaggedTensor attributes from TorchRec
  """

  keys: list[str]
  values: torch.Tensor
  lengths: torch.Tensor


@dataclass
class Batch:
  """
  modified from torchrec.datasets.utils
  """

  dense_features: torch.Tensor
  sparse_features: SparseFeatures
  labels: torch.Tensor

  def to(self, device: str):
    self.dense_features.to(device)
    self.sparse_features.values = self.sparse_features.values.to(device)
    self.sparse_features.lengths = self.sparse_features.lengths.to(device)
    self.labels.to(device)


class _RandomRecBatch:
  """
  modified from torchrec.datasets.random
  """

  generator: torch.Generator | None

  def __init__(
    self,
    keys: list[str],
    batch_size: int,
    hash_sizes: list[int],
    ids_per_features: list[int],
    num_dense: int,
    manual_seed: int | None = None,
    num_generated_batches: int = 10,
    num_batches: int | None = None,
    *,
    min_ids_per_features: list[int] | None = None,
    hash_offsets: list[int] | None = None,
  ) -> None:
    self.keys = keys
    self.keys_length: int = len(keys)
    self.batch_size = batch_size
    self.hash_sizes = hash_sizes
    self.ids_per_features = ids_per_features
    self.min_ids_per_features: list[int] = (
      min_ids_per_features if min_ids_per_features else [0] * len(ids_per_features)
    )
    self.num_dense = num_dense
    self.num_batches = num_batches
    self.num_generated_batches = num_generated_batches

    if hash_offsets:
      assert self.keys_length == len(hash_offsets), (
        "length of hash_offsets must be equal to the number of keys"
      )
    self.hash_offsets = hash_offsets

    if manual_seed is not None:
      self.generator = torch.Generator()
      self.generator.manual_seed(manual_seed)
    else:
      self.generator = None

    self._generated_batches: list[Batch] = [
      self._generate_batch() for _ in range(num_generated_batches)
    ]
    self.batch_index = 0

  def __iter__(self) -> "_RandomRecBatch":
    self.batch_index = 0
    return self

  def __next__(self) -> Batch:
    if self.batch_index == self.num_batches:
      raise StopIteration
    if self.num_generated_batches >= 0:
      batch = self._generated_batches[self.batch_index % len(self._generated_batches)]
    else:
      batch = self._generate_batch()
    self.batch_index += 1
    return batch

  def _generate_batch(self) -> Batch:
    values = []
    lengths = []
    for key_idx, _ in enumerate(self.keys):
      hash_size = self.hash_sizes[key_idx]
      min_num_ids = self.min_ids_per_features[key_idx]
      max_num_ids = self.ids_per_features[key_idx]
      length = torch.randint(
        min_num_ids,
        max_num_ids + 1,
        (self.batch_size,),
        dtype=torch.int32,
        generator=self.generator,
      )
      value = torch.randint(
        0, hash_size, (int(length.sum().item()),), generator=self.generator
      )
      if self.hash_offsets:
        value = value + self.hash_offsets[key_idx]
      lengths.append(length)
      values.append(value)

    sparse_features = SparseFeatures(
      keys=self.keys,
      values=torch.cat(values),
      lengths=torch.cat(lengths),
    )

    dense_features = torch.randn(
      self.batch_size,
      self.num_dense,
      generator=self.generator,
    )
    labels = torch.randint(
      low=0,
      high=2,
      size=(self.batch_size,),
      generator=self.generator,
    )

    batch = Batch(
      dense_features=dense_features,
      sparse_features=sparse_features,
      labels=labels,
    )
    return batch


class RandomRecDataset(IterableDataset[Batch]):
  """
  Random iterable dataset used to generate batches for recommender systems
  (RecSys). Currently produces unweighted sparse features only. TODO: Add
  weighted sparse features.

  Args:
      keys (List[str]): List of feature names for sparse features.
      batch_size (int): batch size.
      hash_size (Optional[int]): Max sparse id value. All sparse IDs will be taken
          modulo this value.
      hash_sizes (Optional[List[int]]): Max sparse id value per feature in keys. Each
          sparse ID will be taken modulo the corresponding value from this argument. Note, if this is used, hash_size will be ignored.
      ids_per_feature (Optional[int]): Number of IDs per sparse feature per sample.
      ids_per_features (Optional[List[int]]): Number of IDs per sparse feature per sample in each key. Note, if this is used, ids_per_feature will be ignored.
      num_dense (int): Number of dense features.
      manual_seed (int): Seed for deterministic behavior.
      num_batches: (Optional[int]): Num batches to generate before raising StopIteration
      num_generated_batches int: Num batches to cache. If num_batches > num_generated batches, then we will cycle to the first generated batch.
                                 If this value is negative, batches will be generated on the fly.
      min_ids_per_feature (Optional[int]): Minimum number of IDs per features.
      min_ids_per_features (Optional[List[int]]): Minimum number of IDs per sparse feature per sample in each key. Note, if this is used, min_ids_per_feature will be ignored.

  Example::

      dataset = RandomRecDataset(
          keys=["feat1", "feat2"],
          batch_size=16,
          hash_size=100_000,
          ids_per_feature=1,
          num_dense=13,
      ),
      example = next(iter(dataset))
  """

  def __init__(
    self,
    keys: list[str],
    batch_size: int,
    hash_size: int | None = None,
    hash_sizes: list[int] | None = None,
    ids_per_feature: int | None = None,
    ids_per_features: list[int] | None = None,
    num_dense: int = 50,
    manual_seed: int | None = None,
    num_batches: int = 1,
    num_generated_batches: int = 50,
    min_ids_per_feature: int | None = None,
    min_ids_per_features: list[int] | None = None,
    stack_feature_ids: bool = False,
  ) -> None:
    super().__init__()

    if hash_sizes is None:
      hash_sizes = [hash_size if hash_size else 100] * len(keys)

    assert hash_sizes is not None
    assert len(hash_sizes) == len(keys), (
      "length of hash_sizes must be equal to the number of keys"
    )

    if ids_per_features is None:
      ids_per_features = [ids_per_feature if ids_per_feature else 2] * len(keys)

    assert ids_per_features is not None
    if min_ids_per_features is None:
      min_ids_per_feature = (
        min_ids_per_feature if min_ids_per_feature is not None else ids_per_feature
      )
      min_ids_per_features = [min_ids_per_feature if min_ids_per_feature else 0] * len(
        keys
      )

    assert len(ids_per_features) == len(keys), (
      "length of ids_per_features must be equal to the number of keys"
    )

    hash_offsets = (
      np.cumsum([0] + hash_sizes).tolist()[:-1] if stack_feature_ids else None
    )

    self.batch_generator = _RandomRecBatch(
      keys=keys,
      batch_size=batch_size,
      hash_sizes=hash_sizes,
      ids_per_features=ids_per_features,
      num_dense=num_dense,
      manual_seed=manual_seed,
      num_batches=None,
      num_generated_batches=num_generated_batches,
      min_ids_per_features=min_ids_per_features,
      hash_offsets=hash_offsets,
    )
    self.num_batches: int = num_batches

  def __iter__(self) -> Iterator[Batch]:
    return itertools.islice(iter(self.batch_generator), self.num_batches)

  def __len__(self) -> int:
    return self.num_batches


def _get_random_dataloader(
  cfg: DictConfig,
  stage: str,
) -> DataLoader[Batch]:
  attr = f"limit_{stage}_batches"
  num_batches = getattr(cfg, attr)
  if stage in ["val", "test"] and cfg.test_batch_size is not None:
    batch_size = cfg.test_batch_size
  else:
    batch_size = cfg.batch_size
  num_embeddings = sum(cfg.num_embeddings_per_feature)
  total_features = len(cfg.num_embeddings_per_feature)
  feature_names = (
    cfg.feature_names
    if cfg.feature_names
    else [f"f_{idx}" for idx in range(total_features)]
  )

  return DataLoader(
    RandomRecDataset(
      keys=feature_names,
      batch_size=batch_size,
      hash_size=num_embeddings,
      hash_sizes=cfg.num_embeddings_per_feature,
      manual_seed=getattr(cfg, "seed", None),
      ids_per_features=cfg.multi_hot_sizes,
      min_ids_per_features=cfg.min_ids_per_features,
      num_dense=cfg.model.dense_features_num,
      num_batches=num_batches,
      num_generated_batches=cfg.num_generated_batches,
      stack_feature_ids=True,
    ),
    batch_size=None,
    batch_sampler=None,
    pin_memory=cfg.pin_memory,
    num_workers=0,
  )


def get_dataloader(cfg: DictConfig, backend: str, stage: str) -> DataLoader[Batch]:
  """
  Gets desired dataloader from dlrm_main command line options. Currently, this
  function is able to return only a DataLoader wrapped around a RandomRecDataset

  Args:
      args (argparse.Namespace): Command line options supplied to dlrm_main.py's main
          function.
      stage (str): "train", "val", or "test".

  Returns:
      dataloader (DataLoader): PyTorch dataloader for the specified options.
  """

  if (
    cfg.in_memory_binary_criteo_path is None
    and cfg.synthetic_multi_hot_criteo_path is None
  ):
    return _get_random_dataloader(cfg, stage)
  else:
    raise NotImplementedError("InMemory criteo dataloader is not implemented yet")
