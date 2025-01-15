# torchax models

## Llama

The `llama` directory demonstrates training of llama model. The model
implementation is forked at the official reference implementation at
https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py
this file is kept as `model_original.py` for reference.

Then, 3 versions of the model is provided: 

* `model.py` is a modification to `model_original.py` by replacing fairscale layers
with regular `torch.nn` layers.

* `model_with_scan.py` is a modification on `model.py` that uses `scan` to iterate
  the layers of model

* `model_with_collectives.py` is a further modification on `model_with_scan.py` by
  using manual collectives (`all_gather`, `all_reduce`) along with `shard_map` to simulate
  how PyTorch native on multi-GPUs would have done. Instead of relying on GSPMD.

Choosing which version of the model to run can be controlled by passing
`--model_impl` flag to the `run.py` script.


## Install dependencies

First one need a valid installation of `torch_xla2`, torch, and jax

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/pytorch/xla.git
cd xla/experimental/torch_xla2
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install optax fire tensorflow tensorboard-plugin-profile
pip install -e .[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

## Running locally

```bash
python run.py --model_impl=<orig|scan|scan_manual>
```

## Run on XPK

Run
`CLUSTER_NAME=... PROJECT_ID=.. ZONE=... launcher/run_xpk.sh <local command>`
will build the dockerfile and launch it on XPK.
Replace `CLUSTER_NAME`, `PROJECT_ID`, and `ZONE` with appropiate values.


## Benchmarks (WIP)

|device| Model size | Batch size | seq length | step time | MFU | NOTEs|
|-------| ----- | ----- | ----- | ----- | ----- | ---|
|TPU v6e-8| 8B |        8 |      8192 | 1.7s | 30% | Scan, fsdp, host-offload|
|TPU v6e-256 x 2| 405B | 256 | 8192 | 46.12s | 28.7% | Scan, fsdp + tp, host-offload|

<!-- TODO: support specifying different XLA flags -->

Llama 3.1 405B on v6e-256 x 2 command:

```sh
NUM_SLICES=2 TPU_TYPE=v6e-256 launcher/run_xpk.sh \
    torchprime/experimental/torchax_models/run.py \
    --batch_size=256 \
    --model_type=405B \
    --seqlen=8192 \
    --use_custom_offload=True \
    --use_custom_mesh=True \
    --model_impl=scan \
    --tp=4 \
    --unroll_layers=1
```
