
# Context Parallelism (CP)

**Context Parallelism (CP)** is a memory-saving distributed training technique designed for models with very long input sequences. It works by sharding the **sequence dimension** of activations and inputs across multiple devices.

This approach is an alternative to Fully Sharded Data Parallelism (FSDP), which shards the *batch dimension*. By sharding the sequence length, CP allows for a much smaller memory footprint per device, enabling training with extremely long contexts even when the per-device batch size is small (e.g., less than 1).

---

## 🧠 Why Use Context Parallelism?

CP is essential when facing two primary constraints in large-scale model training:

* **Memory Limitations**: For models processing tens or hundreds of thousands of tokens, the activation memory can become a bottleneck. CP directly reduces this by distributing the sequence across devices, ensuring the activations fit in memory.
* **Optimal Global Batch Size**: State-of-the-art models are often trained with a specific global batch size (measured in tokens) to achieve optimal convergence. For example, Llama 3 used a 16M token batch. When sequence lengths are very large, CP allows you to maintain a small per-device batch size to stay under a target global token count without sacrificing context length.

---

## ⚖️ How It Works

Sharding the sequence dimension introduces unique challenges, particularly within the attention mechanism.

### The Attention Challenge: Queries, Keys, and Values

In a standard attention operation, every query token must attend to every key/value token. When we shard the sequence dimension:
1.  The **queries (Q)** are sharded along the sequence dimension and remain local to each device.
2.  The **keys (K)** and **values (V)** must be gathered from all other devices (`all-gather`) before the attention computation can be performed.

While this `all-gather` operation introduces communication overhead, it is a necessary step to compute the full attention score.

### The Load Balancing Challenge: Causal Mask Imbalance

Naively sharding a sequence into contiguous chunks (e.g., Device 1 gets tokens 0-1023, Device 2 gets 1024-2047, etc.) creates a computational imbalance. Due to the lower-triangular causal mask in decoder models, later chunks in the sequence perform significantly more computation than earlier ones.



This means devices assigned to later chunks become a bottleneck, slowing down the entire training step.

### Solution: Striped Sharding

To ensure a balanced workload, we implement **striped sharding**. The input sequence is first broken into many small chunks and then distributed in a "striped" pattern across devices.

For example, with 4 devices and 8 chunks of data `[0, 1, 2, 3, 4, 5, 6, 7]`, the distribution would be:
* **Device 1**: Gets chunks `[0, 7]`
* **Device 2**: Gets chunks `[1, 6]`
* **Device 3**: Gets chunks `[2, 5]`
* **Device 4**: Gets chunks `[3, 4]`



This pairing of early and late sequence chunks ensures that each device has a roughly equal computational load. This striping is performed once on the initial input data, adding minimal overhead.

---

## 🚀 Performance & Overhead

The primary communication costs in CP are the same as in FSDP (all-gathering weights and synchronizing gradients). The main *additional* cost is the all-gather operation for the Key and Value caches in each attention layer.

### KV Cache All-Gather Cost

Assuming Grouped-Query Attention (GQA), the ratio of additional computation to communication can be analyzed as follows:

* **Compute (Local Attention)**: $$O(4 \cdot \text{batch} \cdot \frac{\text{seq\_len}^2}{\|CP\|} \cdot \text{query\_heads} \cdot \text{head\_dim})$$
* **Communication (KV All-Gather)**:$$O(4 \cdot \text{batch} \cdot \text{seq\_len} \cdot \text{kv\_heads} \cdot \text{head\_dim})$$
* **Compute-to-Comm Ratio**:$$ \frac{\text{seq\_len} \cdot \text{query\_heads}}{\text{kv\_heads} \cdot \|CP\|} $$

Where:
* `seq_len`: The full sequence length.
* `query_heads`: Number of query heads.
* `kv_heads`: Number of key/value heads.
* `|CP|`: The number of devices in the context parallelism group.

This ratio shows that for long sequences (`seq_len`), the computational benefit significantly outweighs the communication cost, making this a highly effective trade-off.

### Implementation Note

This implementation is a practical and performant flavor of Context Parallelism. More advanced techniques like Ring Attention theoretically hide all communication costs, but come with significantly higher implementation complexity. We find this all-gather approach offers an excellent balance of performance and simplicity.

---

## How to Use

To use context parallelism locally, below is sample command:
(Due to [issue][issue-data-loader], torch-xla parallel-dataloader has some bug in applying sharding on the sequence dimension of inputs. To try out cp, please refer to [this][cp-test])

```sh
python3 torchprime/torch_xla_models/train.py     model=llama-3-8b     task=train     dataset=wikitext     task.global_batch_size=2     ici_mesh.fsdp=4    ici_mesh.context=2
```

To enable load-balanced Context Parallelism, simply set the `load_balance_cp` flag in your model configuration file.

```yaml
# In your model config file
# ...
load_balance_cp: True
# ...



<!-- xrefs -->

[cp-guide]: https://insujang.github.io/2024-09-20/introducing-context-parallelism/
[cp-hf-guide]: https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=context_parallelism
[issue-data-loader]: https://github.com/AI-Hypercomputer/torchprime/issues/353
[cp-test]: https://github.com/AI-Hypercomputer/torchprime/blob/460f54266b89ce09f0497929ad728be2fa32cb18/torchprime/torch_xla_models/tests/test_spmd.py#L134
[lbcp-test]: https://github.com/AI-Hypercomputer/torchprime/blob/460f54266b89ce09f0497929ad728be2fa32cb18/torchprime/torch_xla_models/tests/test_spmd.py#L228C7-L228C37