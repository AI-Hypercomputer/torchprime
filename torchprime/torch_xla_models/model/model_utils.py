"""Utility function(s) for model initialization."""

import importlib
import sys


def initialize_model_class(model_config):
  """Import and initialize model_class specified by the config."""
  full_model_class_string = model_config.model_class
  module_name, model_class_name = full_model_class_string.rsplit(".", 1)

  for candidate_module_name in [f"model.{module_name}", module_name]:
    try:
      module = importlib.import_module(candidate_module_name)
      break
    except ModuleNotFoundError:
      module = None

  if module is None:
    print(f"Error: Failed to import module '{module_name}' or 'model.{module_name}'")
    sys.exit(1)

  if not hasattr(module, model_class_name):
    print(f"Error: Class '{model_class_name}' not found in module '{module.__name__}'")
    sys.exit(1)

  model_class = getattr(module, model_class_name)
  return model_class(model_config)
