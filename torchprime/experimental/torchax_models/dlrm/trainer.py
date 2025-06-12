"""
Minimalistic Torchax based trainer class that is to hide all jax interop calls and should
look like a generic native pt trainer
"""

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import optax
import torch
from omegaconf import DictConfig
from torchax import interop, train


class TorchaxTrainer(ABC):
  def __init__(
    self,
    cfg: DictConfig,
    model: torch.nn.Module,
    loss_fn: Callable,
  ):
    self.config = cfg
    self.device = "jax"

    # TODO: init from config
    # self.jax_optimizer = optax.adagrad(0.006)
    self.jax_optimizer = optax.sgd(cfg.learning_rate)

    model.to(self.device)
    self.model = model

    self.loss_fn = loss_fn

    model_jittable = interop.JittableModule(self.model)
    self.weights = model_jittable.params
    self.buffers = model_jittable.buffers
    self.opt_state = interop.call_jax(self.jax_optimizer.init, self.weights)
    self.model_fn = functools.partial(model_jittable.functional_call, "forward")
    train_step = train.make_train_step(self.model_fn, loss_fn, self.jax_optimizer)
    self.train_step_fn = interop.jax_jit(
      train_step, kwargs_for_jax_jit={"donate_argnums": (0, 2)}
    )

    # TODO: setup scheduler
    # self.scheduler = optax.warmup_cosine_decay_schedule()

  @abstractmethod
  def prepare_module_input(self, batch_data: Any) -> Any:
    pass

  def train_step(self, data: Any) -> Any:
    inputs = self.prepare_module_input(data)
    loss, weights, opt_state = self.train_step_fn(
      self.weights, self.buffers, self.opt_state, inputs, data.labels
    )
    self.weights = weights
    self.opt_state = opt_state
    return loss
