# torch_xla models

## Features

- Optimized for PyTorch/XLA
- Demonstrates GSPMD parallelism
- Supports large language models tasks

## Running locally

1. Clone the repository:

   ```
   git clone https://github.com/AI-Hypercomputer/torchprime.git
   cd torchprime
   ```

2. Install the package:

   ```
   pip install -e .
   ```

3. Run the training script:

   ```
   XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python3 torchprime/torch_xla_models/train.py \
       torchprime/torch_xla_models/configs/run.json
   ```

## Running on XPK

Replace `HF_TOKEN`, `CLUSTER_NAME`, `PROJECT_ID`, and `ZONE` with appropriate
values.

```sh
export HF_TOKEN='... hugging face token ...'
export CLUSTER_NAME='...'
export PROJECT_ID='...'
export ZONE='...'

NUM_SLICES=1 TPU_TYPE=v6e-256 launcher/run_xpk.sh \
   torchprime/torch_xla_models/train.py \
   --dataset_name wikitext \
   --dataset_config_name 'wikitext-2-raw-v1' \
   --output_dir /tmp \
   --cache_dir /tmp \
   --global_batch_size 256 \
   --logging_steps 10 \
   --max_steps 15 \
   --profile_step 5 \
   --model_id 'meta-llama/Meta-Llama-3-8B' \
   --tokenizer_name 'meta-llama/Meta-Llama-3-8B' \
   --block_size 8192 \
   --fsdp full_shard \
   --fsdp_config torchprime/torch_xla_models/configs/fsdp_config.json
```

This will build the dockerfile and launch it on XPK.


## Key Components

- `train.py`: Main training script that sets up the model, data, and training loop
- `configs/run.json`: Configuration file for the training script
- `llama/model.py`: Implementation of the Llama model
