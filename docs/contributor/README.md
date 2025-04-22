# Contributing to torchprime

When developing, use `pip install -e '.[dev]'` to install dev dependencies such
as linter and formatter.

## How to run tests

```sh
pytest
```

## How to run some of the tests, and re-run them whenever you change a file

```sh
tp -i test ... # replace with path to tests/directories
```

## How to format

```sh
ruff format
```

## How to lint

```sh
ruff check [--fix]
```

You can install a Ruff VSCode plugin to check errors and format files from the
editor.

## How to run inside the docker container locally

You can also run locally without XPK with docker. When running inside the docker
container, it will use the same dependencies and build process as used in the
XPK approach, improving the hermeticity and reliability.

```sh
tp docker-run torchprime/torch_xla_models/train.py
```

This will run the torchprime docker image locally. You can also add `--use-hf`
to run HuggingFace model locally.

```sh
tp docker-run --use-hf torchprime/hf_models/train.py
```

## Run distributed training with local torch/torch_xla wheel

torchprime supports running with user specified torch and torch_xla wheels
placed under `local_dist/` directory. The wheel will be automatically installed
in the docker image when use `tp run` command. To use the wheel, add flag
`--use-local-wheel` to `tp run` command:

```sh
tp run --use-local-wheel torchprime/hf_models/train.py
```

The wheels should be built inside a [PyTorch/XLA development docker
image][torch_xla_dev_docker] or the PyTorch/XLA VSCode Dev Container to minimize
compatibility issues.
