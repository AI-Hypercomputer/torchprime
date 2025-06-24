from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "ft"  # Choose the model type: "ft" for fine-tuned, "base" for base model


if MODEL == "ft":
  model_path = "/home/jialeic_google_com/work/torchprime/outputs/export"
  model_id = "meta-llama/Meta-Llama-3-8B"
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  tokenizer.save_pretrained(model_path)

  llm = LLM(model=model_path, dtype="bfloat16")
elif MODEL == "base":
  base_model_path = "/home/jialeic_google_com/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"
  llm = LLM(model=base_model_path, dtype="bfloat16")
else:
  raise ValueError(
    "Invalid model type. Choose 'ft' for fine-tuned or 'base' for base model."
  )

# Define the prompt and sampling parameters
prompt = "What is the capital of France?"
prompt_gsm8k = "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
prompt_gsm8k = f"Question:\n{prompt_gsm8k}\n\n\nAnswer:\n"
sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

# Run inference
outputs = llm.generate([prompt, prompt_gsm8k], sampling_params)

# Print results
for output in outputs:
  print(output.outputs[0].text.strip())
