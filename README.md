# TorchPrime

**Note: this repository is under experimental status. We don't yet have correctness
guarantees for the model implementations.**

TorchPrime is a reference model implementation for PyTorch on TPU/GPU using
`torch_xla` and `torchax`. It is designed to showcase best practices for
high-performance model training with these frameworks.

## Examples

Here is a simple example of training on a single TPU VM. It assumes that you
have already installed torch_xla [1] and torchax [2] following their respective
project READMEs.

Install `torchprime`:

```sh
git clone https://github.com/AI-Hypercomputer/torchprime.git
cd torchprime
pip install -e '.[dev]'
```

### Local training

Train Llama 3 8B using torch_xla:

```sh
export HF_TOKEN='...your huggingface token...'
XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python3 torchprime/torch_xla_models/train.py
```

Train Llama 3 8B using torchax:

```sh
python3 torchprime/experimental/torchax_models/run.py --batch_size=16
```

Refer to `README.md` in `torchprime/torch_xla_models` and
`torchprime/experimental/torchax_models` for more details.

### Distributed training

torchprime uses [xpk][xpk] as the standard path for iterating on
distributed training code.

First teach torchprime about the XPK cluster it is using, the artifact
storage location, etc. You only need to do this on first clone or when
switching to a different topology or cluster. Example:

```sh
tp use \
    --cluster <XPK CLUSTER NAME> \
    --project my-gcp-project \
    --zone us-east5-b \
    --num-slices 1 \
    --tpu-type v6e-256 \
    --artifact-dir gs://bucket/dir
```

Then prepend `tp run` to a particular Python file you would like to
run remotely, including arguments, e.g.

```sh
tp run torchprime/experimental/torchax_models/run.py --batch_size=256
```

`tp run` will broadcast this command to all VMs in the XPK cluster,
which is the convention for running SPMD distributed workloads.

## Structure

This repo will contain a set of reference models that we have optimized and
runs well on TPU. The best performing scaling configuration
(parallelism techniques, checkpointing, etc.) for a model on various hardwares
will be provided for ease of reproducibility.

`docs` contains guides for optimizing performance and debugging issues.

`torchprime/launcher` contains scripts to train a model on a large TPU cluster.

`torchprime/data` contains dataset and data loading utilities.

`torchprime/torch_xla_models` contains model implementations using `torch_xla`.

`torchprime/experimental/torchax_models` contains model implementations using
`torchax`.

Finally, each model may also provide a GPU "original" version that illustrates
and attributes where this model code came from, if any. This also helps to
show case what changes we have done to make it performant on TPU. The original
version is not expected to be run.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

When developing, use `pip install -e '.[dev]'` to install dev dependencies such
as linter and formatter.

How to run tests:

```sh
pytest
```

How to run some of the tests, and re-run them whenever you change a file:

```sh
tp -i test ... # replace with path to tests/directories
```

How to format:

```sh
ruff format
```

How to lint:

```sh
ruff check [--fix]
```

You can install a Ruff VSCode plugin to check errors and format files from
the editor.

## License

This project is licensed under the New BSD License - see the [LICENSE](LICENSE)
file for details.

For more information on PyTorch/XLA, visit the
[official documentation](https://github.com/pytorch/xla).

[1]: https://github.com/pytorch/xla
[2]: https://github.com/pytorch/xla/tree/master/torchax
[xpk]: https://github.com/AI-Hypercomputer/xpk
