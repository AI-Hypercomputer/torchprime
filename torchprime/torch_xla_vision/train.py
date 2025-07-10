"""Trainers for model."""

import logging
import os
import pprint
import sys

import hydra
import torch
import torcheval.metrics
import transformers
from omegaconf import DictConfig
from tqdm import tqdm

from torchprime import models
from torchprime.torch_xla_vision import data, metrics_log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_dataloaders(
  train_batch_size: int,
  test_batch_size: int,
  num_workers: int,
  label_attribute: str,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, list[int]]:
  """
  Initializes and returns data loaders for the training and test sets.

  Args:
      train_batch_size: The batch size for the training DataLoader.
      test_batch_size: The batch size for the test DataLoader.
      num_workers: The number of worker processes for data loading.
      label_attribute: The attribute to use for classification.

  Returns:
      A tuple containing the training DataLoader, test DataLoader, and a list of
      class labels.
  """
  train_ds, test_ds = data.get_splits(label_attribute=label_attribute)

  train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=True,
    drop_last=True,
  )
  test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False,
    prefetch_factor=2,
    persistent_workers=True,
    drop_last=False,
  )
  return train_loader, test_loader, train_ds.classes


def _get_model(
  model_id: str, num_classes: int, num_channels: int, seed: int, device: torch.device
) -> tuple[torch.nn.Module, str]:
  """
  Initializes and returns the model and its hash.

  Args:
      model_id: The identifier for the model in the model registry.
      num_classes: The number of output classes for the model.
      num_channels: The number of input channels for the model.
      seed: The random seed for model initialization.
      device: The device to move the model to.

  Returns:
      A tuple containing the initialized model and its state hash.
  """
  model_factory = models.registry.get(model_id)
  assert model_factory is not None, f"Model with id {model_id} not registered"
  net, model_hash = model_factory(
    num_classes=num_classes,
    num_channels=num_channels,
    seed=seed,
  )
  return net.to(device), model_hash


def train_step(net, image, label, criterion, optimizer):
  """
  Performs a single training step for the model.

  This includes zeroing gradients, performing a forward pass, calculating the
  loss, performing a backward pass, and updating the model parameters.

  Args:
      net: The neural network model.
      image: The input image tensor.
      label: The ground truth label tensor.
      criterion: The loss function.
      optimizer: The optimizer.

  Returns:
      The loss tensor for the step.
  """
  # Zero the parameter gradients
  optimizer.zero_grad()

  # Forward pass
  output = net(image)
  loss = criterion(output, label)

  # Backward pass and optimize
  loss.backward()
  optimizer.step()

  return loss


def eval(net, loader, criterion, metrics, device) -> list[float]:
  """
  Evaluates the model on a given dataset.

  Args:
      net: The neural network model.
      loader: The DataLoader for the evaluation dataset.
      criterion: The loss function.
      metrics: A dictionary of torcheval metrics to compute.
      device: The device to run evaluation on.

  Returns:
      A list of loss values for each batch in the loader.
  """
  net.eval()

  with torch.no_grad():
    losses = []
    for batch in loader:
      image, label = batch[0].to(device), batch[1].to(device)
      output = net(image)
      loss = criterion(output, label)
      losses.append(loss.item())  # Barrier - obviates mark_step.
      for metric in metrics.values():
        metric.update(output, label)

  return losses  # Metrics are updated in place


def log_stats(epoch, losses, net, test, criterion, metrics, device, metrics_logger):
  """
  Computes, logs, and prints training and evaluation statistics for an epoch.

  Args:
      epoch: The current epoch number.
      losses: A list of training losses from the epoch.
      net: The neural network model.
      test: The DataLoader for the test set.
      criterion: The loss function.
      metrics: A dictionary of torcheval metrics to compute.
      device: The device to run evaluation on.
      metrics_logger: The logger instance for saving metrics to a file.
  """
  metrics_to_log = {}

  logger.info(f"Epoch {epoch}")
  train_loss = sum(losses) / len(losses)
  metrics_to_log["train_loss"] = f"{train_loss:.4f}"

  test_losses = eval(net, test, criterion, metrics, device)
  test_loss = sum(test_losses) / len(test_losses)
  metrics_to_log["test_loss"] = f"{test_loss:.4f}"
  for name, metric in metrics.items():
    metrics_to_log[name] = f"{metric.compute().item():.4f}"
    metric.reset()

  logger.info(pprint.pformat(metrics_to_log))
  metrics_logger.log(epoch, metrics_to_log)


def trainer(
  config: DictConfig,
  device: torch.device,
  compile_fn,
):
  """
  A binary image classification trainer with AdamW, LR scheduling, and a
  fine-tuning strategy (freezing/unfreezing the backbone).

  Args:
      config: The hydra configuration.
      device: The device to train on (e.g., 'cuda', 'xla').
      compile_fn: The function to compile the training step (e.g., torch.compile).
  """
  metrics_logger = metrics_log.MetricsLogger(
    f"prod_{config.seed}",
    {"lr": config.lr, "device": device.type, "model": config.model_id},
  )

  torch.manual_seed(config.seed)

  train, test, classes = _get_dataloaders(
    config.train_batch_size,
    config.test_batch_size,
    config.num_workers,
    label_attribute=config.label_attribute,
  )

  num_channels = train.dataset[0][0].shape[0]
  net, _ = _get_model(config.model_id, len(classes), num_channels, config.seed, device)

  net.freeze_backbone()

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=0.001)

  lr_scheduler = transformers.get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train) * 1,
    num_training_steps=len(train) * 5,
  )

  train_step_compiled = compile_fn(train_step)

  metrics = {
    "accuracy_top1": torcheval.metrics.MulticlassAccuracy(num_classes=len(classes), k=1)
  }

  for epoch in range(1, config.epochs + 1):
    net.train()

    losses = []

    use_tqdm = sys.stdout.isatty()
    pbar = tqdm(train, disable=not use_tqdm, desc=f"Epoch {epoch}")
    for batch in pbar:
      inputs, labels = batch[0].to(device), batch[1].to(device)
      loss = train_step_compiled(net, inputs, labels, criterion, optimizer)
      losses.append(loss.item())
      lr_scheduler.step()
      pbar.set_postfix({"loss": loss.item()})

    if epoch == 5:
      net.unfreeze_all()
      optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=0.001)
      lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train) * 1,
        num_training_steps=len(train) * (config.epochs - 5),
      )

    log_stats(epoch, losses, net, test, criterion, metrics, device, metrics_logger)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: DictConfig):
  """
  Main entry point for training a model on the CelebA dataset.

  Sets up the device and compilation function, then calls the appropriate
  training function.

  Args:
      config: The hydra configuration.
  """
  if torch.cuda.is_available():
    device = torch.device("cuda")
    if config.use_torch_compile:
      logger.info("Using torch.compile on CUDA.")
      compile_fn = torch.compile
    else:

      def compile_fn(fn):
        return fn

  else:
    os.environ["XLA_COMPILE_CACHE_PATH"] = "/tmp/xla_cache"

    import torch_xla

    device = torch.device("xla")
    compile_fn = torch_xla.compile

  logger.info(
    f"Running training for model {config.model_id} with lr={config.lr}, seed={config.seed}, device={device}"
  )

  trainer(config, device, compile_fn)


if __name__ == "__main__":
  main()
