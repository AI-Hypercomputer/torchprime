"""Durable storage for metrics from experiements."""

import csv
import datetime
import pprint


class MetricsLogger:
  def __init__(self, run_name, hyperparameters: dict, path="metrics.csv"):
    self.path = path
    self.datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    self.run_name = run_name
    self.hyperparameters = pprint.pformat(hyperparameters)  # Remember to escape this

  def log(self, epoch, metrics: dict):
    hyperparameters_str = pprint.pformat(self.hyperparameters)
    with open(self.path, "a") as f:
      writer = csv.writer(f)
      if f.tell() == 0:
        writer.writerow(
          ["run_name", "datetime", "hyperparameters", "epoch"] + list(metrics.keys())
        )
      writer.writerow(
        [self.run_name, self.datetime, hyperparameters_str, epoch]
        + list(metrics.values())
      )
