"""
Torchax based trainin pipeline for DLRM model
"""

import logging
import time
from collections.abc import Callable

import hydra
import jax
import torch
import torchax as tx
from omegaconf import DictConfig, OmegaConf

from torchprime.experimental.torchax_models.dlrm.data import Batch, get_dataloader
from torchprime.experimental.torchax_models.dlrm.model import DlrmModel
from torchprime.experimental.torchax_models.dlrm.trainer import TorchaxTrainer

LossType = float

logger = logging.getLogger(__name__)

class Trainer(TorchaxTrainer):
  def __init__(
    self,
    cfg: DictConfig,
    model: torch.nn.Module,
    loss_fn: Callable,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
  ):
    super().__init__(cfg, model, loss_fn)
    self.train_dataloader = train_dataloader
    self.test_dataloader = test_dataloader
    self.val_dataloader = val_dataloader

  def run_train(self):
    epochs = self.config.train_epochs
    for epoch in range(epochs):
      logger.info(f"EPOCH {epoch + 1}:")
      _epoch_loss = self.run_train_epoch(epoch)
      # TODO: validate

  def run_train_epoch(self, epoch: int) -> LossType:
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    epoch_start_time = time.time()
    for i, data in enumerate(self.train_dataloader):
      step_start_time = time.time()
      loss = self.train_step(data)
      # Gather data and report
      running_loss += loss.item()

      # TODO: report every N steps from cfg
      if i % 1 == 0:
        last_loss = running_loss  # loss per batch
        logger.info(
          f"  batch {i + 1} loss: {last_loss}"
          + f" step time = {time.time() - step_start_time}"
        )
        running_loss = 0.0

    logger.info(f" epoch time = {time.time() - epoch_start_time}")
    return last_loss

  def prepare_module_input(self, batch_data: Batch):
    batch_data.to(self.device)
    return (
      batch_data.dense_features,
      batch_data.sparse_features.values,
      batch_data.sparse_features.lengths,
    )

  # TODO:
  def run_validate(self):
    pass

  # TODO:
  def run_test(self):
    pass


@hydra.main(version_base=None, config_path="configs", config_name="dlrm")
def main(cfg: DictConfig):
  logger.info(OmegaConf.to_yaml(cfg))  # Print the config for debugging
  logger.info(locals())
  torch.manual_seed(0)
  torch.set_default_dtype(torch.bfloat16)
  tx.enable_performance_mode()
  tx.enable_globally()

  logger.info("Local devices num:", jax.local_device_count())
  device = "jax"

  env = tx.default_env()
  env.config.use_torch_native_for_cpu_tensor = False

  train_dataloader = get_dataloader(cfg, device, "train")
  val_dataloader = get_dataloader(cfg, device, "val")
  test_dataloader = get_dataloader(cfg, device, "test")

  model = DlrmModel.from_cfg(cfg)
  loss_fn = torch.nn.BCEWithLogitsLoss()

  trainer = Trainer(
    cfg, model, loss_fn, train_dataloader, test_dataloader, val_dataloader
  )
  trainer.run_train()


if __name__ == "__main__":
  main()
