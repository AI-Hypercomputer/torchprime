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
