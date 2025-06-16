import torch
from transformers import (
  AutoModelForCausalLM,
  AutoTokenizer,
  Trainer,
  TrainingArguments,
  set_seed,
)

from torchprime.data import make_sft_dataset

set_seed(42)  # or any integer
torch.manual_seed(42)

# model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
  model_name,
  torch_dtype=torch.bfloat16,
)

# Enable memory optimizations
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# dataset
dataset_config = {
  "hf_dataset_name": "gsm8k",
  "hf_dataset_config_name": "main",
  "split": "train",
  "block_size": 256,
  "cache_dir": "/tmp/",
  "format": "prompt_completion",
  "compute_loss_on": "completion",
  "pack_samples": False,
  "truncation": "drop",
}
dataset = make_sft_dataset(**dataset_config, tokenizer=tokenizer)

assert all(len(x["input_ids"]) == dataset_config["block_size"] for x in dataset), (
  "Padding is inconsistent"
)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

tokenizer.pad_token = tokenizer.eos_token

# Training args with FSDP and Adafactor
training_args = TrainingArguments(
  output_dir="/tmp/llama3-gsm8k-fsdp",
  per_device_train_batch_size=16,
  gradient_accumulation_steps=1,
  max_steps=100,
  learning_rate=1e-5,
  max_grad_norm=1.0,
  lr_scheduler_type="linear",
  warmup_steps=10,
  bf16=True,
  optim="adafactor",
  logging_steps=1,
  save_strategy="epoch",
  fsdp="full_shard auto_wrap",
  fsdp_transformer_layer_cls_to_wrap="LlamaDecoderLayer",
  ddp_find_unused_parameters=False,
)


trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=dataset,
  tokenizer=tokenizer,
  # data_collator = stack_collator
)

trainer.train()
