Following https://docs.vllm.ai/en/stable/getting_started/installation/ai_accelerator.html#set-up-using-python

- set up a conda environment
```
conda create -n vllm python=3.10 -y
conda activate vllm
git clone https://github.com/vllm-project/vllm.git && cd vllm
pip install -r requirements/tpu.txt
sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev
VLLM_TARGET_DEVICE="tpu" python -m pip install -e .
```

- Deploy the local checkpoint via vllm and do batch inference
```
python torchprime/personal/jialei/vllm-tpu-deployment/deployment.py
```