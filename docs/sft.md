# Supervised fine-tuning (SFT)

`torchprime` ships with an `SFTTrainer` for supervised fine-tuning tasks. The trainer
loads a pretrained checkpoint before training starts and automatically exports the
final model at the end of training.

## Quick example

Fine-tune Llama 3 8B on the GSM8k dataset using the predefined configuration:

```sh
python3 torchprime/torch_xla_models/train.py --config-name llama-3-8b-sft-w-gsm8k
```

This configuration loads the `meta-llama/Meta-Llama-3-8B` weights, trains on the
GSM8k dataset and saves the resulting checkpoint in the directory specified by
`task.export_checkpoint_path`.

## Custom fine-tuning runs

To use your own dataset or model checkpoint, point the trainer to the SFT configs
and specify the pretrained model:

```sh
python3 torchprime/torch_xla_models/train.py \
    dataset=my_dataset \
    task=sft \
    model.pretrained_model=my/checkpoint
```

See [`configs/dataset/gsm8k.yaml`](../torchprime/torch_xla_models/configs/dataset/gsm8k.yaml)
for an example of how to configure a dataset.

## Deploying an SFT checkpoint with vLLM

Once training completes you can load the exported checkpoint with
[vLLM](https://github.com/vllm-project/vllm) for inference on TPU. The
`torchprime/experimental/vllm_tpu_deployment` directory contains an example
script and setup instructions. After following the steps in
`experimental/vllm_tpu_deployment/README.md`, run:

```sh
python torchprime/experimental/vllm_tpu_deployment/deploy.py
```

The script loads either the fine-tuned weights or the base model and runs a
couple of example prompts.

## (Optional) GPU comparison scripts

The `experimental/torchrun_gpu_sft` folder mirrors the TPU workflow so that
training performance can be compared directly with a standard GPU setup. To run
the GPU example on a multi-GPU machine:

```sh
torchrun --nproc_per_node=4 torchprime/experimental/torchrun_gpu_sft/run_sft_gsm8k.py
```

After the runs finish, update the metrics files in that directory and execute

```sh
python torchprime/experimental/torchrun_gpu_sft/draw_figure.py
```

to plot a side‑by‑side training curve for TPU and GPU.
