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
   XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 python3 torchprime/torch_xla_models/train.py
   ```

## Running on XPK

Follow the guide in `tp use` to setup the cluster information.

```sh
export HF_TOKEN='... hugging face token ...'
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1 

tp run torchprime/torch_xla_models/train.py
```

This will build the dockerfile and launch it on XPK.


## Key Components

- `train.py`: Main training script that sets up the model, data, and training loop
- `configs/base.yaml`: Configuration file for the training script
- `configs/model`: Configuration files for the training models
- `llama/model.py`: Implementation of the Llama model
