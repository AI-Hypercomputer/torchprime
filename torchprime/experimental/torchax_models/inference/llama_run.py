import torchax.interop
from torchprime.experimental.torchax_models.inference.llama_run import model
import torch
import torchax
import torchax.config
import jax
import time

env = torchax.default_env()
torch.manual_seed(42)
torch.set_default_dtype(torch.bfloat16)
torchax.enable_performance_mode()

max_seq_len = 512  # 8192
vocab_size = 128  # 32000
n_layer = 1
n_heads = 4
dim = 8
block_size = 16  # 2048
batch_size = 1


def fake_dataloader(size, vocab_size, seqlen, batch_size):
  for _ in range(size):
    x = torch.randint(0, vocab_size, (batch_size, seqlen), device="cpu")
    yield x


if __name__ == "__main__":
  with torch.no_grad():
    input = torch.randint(0, vocab_size, (1, max_seq_len))
    model_args = model.ModelArgs(
      block_size=block_size,
      vocab_size=vocab_size,
      n_layer=n_layer,
      n_heads=n_heads,
      dim=dim,
      max_seq_len=max_seq_len,
    )
    freqs_cis = model.precompute_freqs_cis(
      model_args.dim // model_args.n_heads,
      model_args.max_seq_len,
      model_args.rope_theta,
      model_args.use_scaled_rope,
    ).to(torch.bfloat16)
    m = model.Transformer(model_args)
    m.to(torch.bfloat16)

    # TODO: move weight as arguemts of forward
    def forward(input, freqs_cis, mask):
      return m(input, 0, freqs_cis=freqs_cis, mask=mask)

    jitted_forward = torchax.interop.jax_jit(forward)

    data_iter = fake_dataloader(5, vocab_size, max_seq_len, batch_size)
    with env:
      m.to("jax")
      freqs_cis = freqs_cis.to("jax")
      for i, input in enumerate(data_iter):
        input = input.to("jax")
        mask = torch.ones_like(input)
        step_start = time.perf_counter()
        output = jitted_forward(input, freqs_cis, mask)
        jax.block_until_ready(torchax.tensor.t2j(output))
        step_end = time.perf_counter()
        print(
          i,
          "step latency: ",
          step_end - step_start,
        )
