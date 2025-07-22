This example shows how to deploy a checkpoint with `vllm` on TPU.
It follows the official vLLM installation instructions and runs a small
batch inference script.

### Environment setup
Follow the [vLLM TPU installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/ai_accelerator.html#set-up-using-python) and create a conda environment:

```
conda create -n vllm python=3.10 -y
conda activate vllm
git clone https://github.com/vllm-project/vllm.git && cd vllm
pip install -r requirements/tpu.txt
sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev
VLLM_TARGET_DEVICE="tpu" python -m pip install -e .
```

### Run inference
After installing vLLM, run:

```
python torchprime/experimental/vllm_tpu_deployment/deploy.py
```

The script loads the model checkpoint and prints the generated outputs for a
couple of example prompts.
