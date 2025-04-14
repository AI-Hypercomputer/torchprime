<div align="center">

# torchprime

#### High-performance training for PyTorch on Cloud TPU

</div>
<br /><br />

torchprime is a reference implementation for training PyTorch models on TPU. It
is designed to showcase best practices for large-scale, high-performance model
training using `torch_xla` ([project][torch_xla]), with
minimal changes to model code. It aims to demystify training on XLA-based
accelerators, providing clear patterns and best practices to help the PyTorch
community unlock top performance and efficiency on Google Cloud TPUs.

torchprime is under active development, and we're eager for feedback and input
from the PyTorch community.

## Installation

Before installing torchprime, you will need to first install
[torch_xla][torch_xla] following its respective project README.

Install `torchprime`:

```sh
git clone https://github.com/AI-Hypercomputer/torchprime.git
cd torchprime
pip install -e '.[dev]'
```

## Examples

### Local training

Here is a simple example of training on a single TPU VM. Train Llama 3 8B using
torch_xla:

```sh
export HF_TOKEN='...your huggingface token...'
XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python3 torchprime/torch_xla_models/train.py
```

Refer to `README.md` in `torchprime/torch_xla_models` for more details.

### Configuring training

torchprime uses [hydra][hydra] to read configurations (e.g. model name, batch
size) from the command line and `.yaml` files.

In the `torch_xla_models` directory, you'll find a `configs/default.yaml`. That
specifies the default configuration for the trainer. You may override configs on
the command line with a `key=value` syntax. For example, the following command
will train Mixtral 8x7B with a global batch size of 256, and set the FSDP SPMD
ICI mesh axis length to 64:

```sh
python3 torchprime/torch_xla_models/train.py \
    model=mixtral-8x7b \
    global_batch_size=256 \
    ici_mesh.fsdp=64
```

You may refer to the hydra docs for other ways to specify configs.

### Distributed training

torchprime uses [xpk][xpk] as the standard path for iterating on distributed
training code.

First teach torchprime about the XPK cluster it is using, the artifact storage
location, etc. You only need to do this on first clone or when switching to a
different topology or cluster. Example:

```sh
tp use \
    --cluster <XPK CLUSTER NAME> \
    --project my-gcp-project \
    --zone us-east5-b \
    --num-slices 1 \
    --tpu-type v6e-256 \
    --artifact-dir gs://bucket/dir
```

torchprime natively supports [multi-slice or multi-pod][multi-slice] training.
`--num-slices` specifies the number of [slices][tpu-slice] used by the workload.
`--tpu-type` specifies the [accelerator type][accelerator-type] in each slice.
To do multi-pod training, simply specify a `--tpu-type` that is as big as a
[pod][tpu-pod].

After configuring the cluster, prepend `tp run` to a particular Python file you
would like to run remotely, including arguments, e.g.

```sh
# Train Llama 3.0 8B on 256 chips
tp run torchprime/torch_xla_models/train.py \
    model=llama-3-8b \
    global_batch_size=256 \
    ici_mesh.fsdp=256
```

`tp run` will broadcast the specified command to all VMs in the XPK cluster,
which is the convention for running SPMD distributed workloads. See `tp run
--help` for more advanced features.

#### Env vars passed to the workload

`tp run` will pick up these environment variables locally and proxy them to the
distributed workload, if found:

- `HF_TOKEN`: HuggingFace token
- `XLA_IR_DEBUG`: [torch_xla debugging flag][torch_xla_debug_env]
- `XLA_HLO_DEBUG`: [torch_xla debugging flag][torch_xla_debug_env]
- `LIBTPU_INIT_ARGS`: XLA flags that affect compilation and execution behavior

#### Additional CLI arguments passed to the workload

Besides forwarding your command line arguments, `tp run` will add:

- `profile_dir=[...]`: path to a [profile][torch_xla_profile] directory
  accessible by the workload
- `output_dir=[...]` path to a directory where the workload may store output
  artifacts such as metrics and checkpoints

## Supported Models

Below are the status of various models. There are five stages for each model:

1. **TODO**: We need to implement the model.
1. **Implemented**: The model runs either a training or an inference step.
1. **Optimized**: We found the best scaling configuration for the model on one
  or more hardware. One-off performance data is available.
1. **Convergence**: We tested that the training loss converges to a reasonable
  value, or that the loss curve tracks an existing reference if exists.
1. **Production**: Not only is the model optimized and converges, its
  performance is also continuously monitored. This is a good state for using the
  model in production.

All implemented models will at least have unit tests to verify basic numerical
correctness, and the convergence verification stage serves as an additional
correctness guarantee.

If a model is implemented, you'll also find a training recipe linked from the
checkmark emoji in the table. If a model is optimized, you'll also find MFU
numbers linked from the table. Note that a model may continue to receive ongoing
optimization thereafter.

| **Model**            | **Implemented**                                                        | **Optimized**                                                        | **Converges** |
| -------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------- |
| Llama 3.0 8B         | [✅](torchprime/torch_xla_models/README.md#llama-30-8b-on-v6e-256)     | [✅](torchprime/torch_xla_models/README.md#llama-30-8b-on-v6e-256)   | [TODO](https://github.com/AI-Hypercomputer/torchprime/issues/90) |
| Llama 3.1 8B         | [✅](torchprime/torch_xla_models/README.md#llama-31-8b-on-v6e-256)     | [TODO](https://github.com/AI-Hypercomputer/torchprime/issues/133)    | TODO |
| Llama 3.1 70B        | [TODO](https://github.com/AI-Hypercomputer/torchprime/issues/17)       | TODO                                                                 | TODO |
| Llama 3.1 405B       | [✅](torchprime/torch_xla_models/README.md#llama-31-405b-on-v6e-256)   | [✅](torchprime/torch_xla_models/README.md#llama-31-405b-on-v6e-256) | TODO |
| Llama 4 Scout        | [TODO](https://github.com/AI-Hypercomputer/torchprime/issues/198)      | TODO | TODO |
| Llama 4 Maverick     | [TODO](https://github.com/AI-Hypercomputer/torchprime/issues/200)      | TODO | TODO |
| Mixtral 8x7B         | [✅](torchprime/torch_xla_models/README.md#mixtral-8x7b-on-v6e-256)    | [TODO](https://github.com/AI-Hypercomputer/torchprime/issues/44)     | TODO |
| Mixtral 8x22B        | [TODO](https://github.com/AI-Hypercomputer/torchprime/issues/45)       | TODO | TODO |
| DeepSeek V3/R1       | TODO                                                                   | TODO | TODO |
| Stable Diffusion 2.0 | [TODO](https://github.com/AI-Hypercomputer/torchprime/issues/87)       | TODO | TODO |
| Stable Diffusion 2.1 | [TODO](https://github.com/AI-Hypercomputer/torchprime/issues/88)       | TODO | TODO |

## Structure

This repo will contain a set of reference models that we have optimized and runs
well on TPU. The best performing scaling configuration (parallelism techniques,
checkpointing, etc.) for a model on various hardwares will be provided for ease
of reproducibility.

`docs` contains guides for optimizing performance and debugging issues.

`torchprime/launcher` contains scripts to train a model on a large TPU cluster.

`torchprime/data` contains dataset and data loading utilities.

`torchprime/torch_xla_models` contains model implementations using `torch_xla`.

`torchprime/experimental/torchax_models` contains model implementations using
`torchax`.

Finally, each model may also provide a GPU "original" version that illustrates
and attributes where this model code came from, if any. This also helps to
showcase what changes we have done to make it performant on TPU. The original
version is not expected to be run.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

When developing, use `pip install -e '.[dev]'` to install dev dependencies such
as linter and formatter.

### How to run tests

```sh
pytest
```

### How to run some of the tests, and re-run them whenever you change a file

```sh
tp -i test ... # replace with path to tests/directories
```

### How to format

```sh
ruff format
```

### How to lint

```sh
ruff check [--fix]
```

You can install a Ruff VSCode plugin to check errors and format files from the
editor.

### How to run inside the docker container locally

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

## License

This project is licensed under the New BSD License - see the [LICENSE](LICENSE)
file for details.

For more information on PyTorch/XLA, visit the [official
documentation](https://github.com/pytorch/xla).

[torch_xla]: https://github.com/pytorch/xla
[xpk]: https://github.com/AI-Hypercomputer/xpk
[torch_xla_debug_env]:
    https://github.com/pytorch/xla/blob/master/docs/source/learn/troubleshoot.md#environment-variables
[torch_xla_profile]:
    https://github.com/pytorch/xla/blob/master/docs/source/learn/troubleshoot.md#performance-profiling
[hydra]: https://hydra.cc/docs/intro/
[torch_xla_dev_docker]:
    https://github.com/pytorch/xla/blob/master/CONTRIBUTING.md#manually-build-in-docker-container
[tpu-pod]: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu-pod
[tpu-slice]: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#slices
[accelerator-type]: https://cloud.google.com/tpu/docs/multislice-introduction#concepts
[multi-slice]: https://cloud.google.com/tpu/docs/multislice-introduction
