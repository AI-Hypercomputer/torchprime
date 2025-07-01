"""Trainers for model."""

import logging
import os
import pprint
import sys

import torch
import torcheval.metrics
import transformers
from tqdm import tqdm

import data
import metrics_log
import model as model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_step(net, image, label, criterion, optimizer):
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
    metrics_to_log = {}

    logger.info(f"Epoch {epoch}")
    train_loss = sum(losses) / len(losses)
    metrics_to_log["train_loss"] = f"{train_loss:.4f}"
    del losses

    test_losses = eval(net, test, criterion, metrics, device)
    test_loss = sum(test_losses) / len(test_losses)
    metrics_to_log["test_loss"] = f"{test_loss:.4f}"
    for name, metric in metrics.items():
        metrics_to_log[name] = f"{metric.compute().item():.4f}"
        metric.reset()

    logger.info(pprint.pformat(metrics_to_log))
    metrics_logger.log(epoch, metrics_to_log)


def train_simple(lr, seed, device, compile_fn, quick=False):
    metrics_logger = metrics_log.MetricsLogger(
        f"simple_{seed}", {"lr": lr, "device": device.type}
    )

    torch.manual_seed(seed)

    train_ds, test_ds = data.get_splits(quick)

    train = torch.utils.data.DataLoader(
        train_ds,
        batch_size=2048,
        shuffle=True,
        num_workers=16,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True,
    )
    test = torch.utils.data.DataLoader(
        test_ds,
        batch_size=4096,
        shuffle=False,
        num_workers=16,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=False,
    )

    net, _ = model.get_model(
        num_classes=len(train_ds.classes),
        num_channels=train_ds[0][0].shape[0],
        seed=seed,
    )
    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    train_step_compiled = compile_fn(train_step)

    metrics = {
        "accuracy_top1": torcheval.metrics.MulticlassAccuracy(
            num_classes=len(train_ds.classes), k=1
        ),
    }

    for epoch in range(1, 26):
        net.train()

        losses = []

        use_tqdm = sys.stdout.isatty()
        pbar = tqdm(train, disable=not use_tqdm)
        for batch in pbar:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            loss = train_step_compiled(net, inputs, labels, criterion, optimizer)
            losses.append(loss.item())
            pbar.set_postfix({"loss": loss})

        # Print statistics
        log_stats(epoch, losses, net, test, criterion, metrics, device, metrics_logger)


def train_prod(lr, seed, device, compile_fn, quick=False):
    metrics_logger = metrics_log.MetricsLogger(
        f"prod_{seed}", {"lr": lr, "device": device.type}
    )

    torch.manual_seed(seed)

    train_ds, test_ds = data.get_splits(quick)

    train = torch.utils.data.DataLoader(
        train_ds,
        batch_size=2048,
        shuffle=True,
        num_workers=16,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True,
    )
    test = torch.utils.data.DataLoader(
        test_ds,
        batch_size=4096,
        shuffle=False,
        num_workers=16,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=False,
    )

    net, _ = model.get_model(
        num_classes=len(train_ds.classes),
        num_channels=train_ds[0][0].shape[0],
        seed=seed,
    )
    net = net.to(device)
    # Prod: Freeze backbone.
    model.freeze_backbone(net)

    criterion = torch.nn.CrossEntropyLoss()
    # Prod: AdamW optimizer with weight decay.
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.001)

    # Prod: Add an LR scheduler
    lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train) * 1,
        num_training_steps=len(train) * 5
    )

    train_step_compiled = compile_fn(train_step)

    metrics = {
        "accuracy_top1": torcheval.metrics.MulticlassAccuracy(
            num_classes=len(train_ds.classes), k=1
        ),
    }

    for epoch in range(1, 26):
        net.train()

        losses = []

        use_tqdm = sys.stdout.isatty()
        pbar = tqdm(train, disable=not use_tqdm)
        for batch in pbar:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            loss = train_step_compiled(net, inputs, labels, criterion, optimizer)
            losses.append(loss)
            # Prod: Step the LR scheduler
            lr_scheduler.step()
            pbar.set_postfix({"loss": loss})
        # Prod: Unfreeze the backbone after a few epochs.
        if epoch == 5:
            model.unfreeze_all(net)
            optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.001)
            lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=len(train) * 1,
                num_training_steps=len(train) * 20)

        # Print statistics
        log_stats(epoch, losses, net, test, criterion, metrics, device, metrics_logger)


def main() -> int:
    arg = sys.argv[1]

    if torch.cuda.is_available():
        device = torch.device("cuda")
        compile_fn = lambda x: x # torch.compile
    else:
        os.environ["XLA_COMPILE_CACHE_PATH"] = "/tmp/xla_cache"

        import torch_xla

        device = torch.device("xla")
        compile_fn = torch_xla.compile

    if arg == "quick":
        logger.info(f"Running quick training with seed {42}, device {device}")
        train_simple(0.1, 42, device, compile_fn, quick=True)
    elif arg == "long":
        logger.info(f"Running long training with seed {42}, device {device}")
        train_simple(0.1, 42, device, compile_fn)
    elif arg == "simple":
        lrs = [10.0, 1.0, 0.1]
        seeds = [42, 42, 0, 1, 2, 3]
        for lr in lrs:
            for seed in seeds:
                logger.info(
                    f"Running simple training with seed {seed}, lr {lr}, device {device}"
                )
                train_simple(lr, seed, device, compile_fn)
    elif arg == "prod":
        lrs = [0.003, 0.0003]
        seeds = [42, 42, 0, 1, 2, 3]
        for lr in lrs:
            for seed in seeds:
                logger.info(
                    f"Running prod training with seed {seed}, lr {lr}, device {device}"
                )
                train_prod(lr, seed, device, compile_fn)
    else:
        logger.error(f"Unknown argument: {arg}. Pass simple or prod as the first arg.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
