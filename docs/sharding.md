# How to configure model sharding

Here is guide for how to shard models (i.e. apply N-dimensional parallelism)
in torchprime.

Since torchprime uses the SPMD paradigm, we recommend familiarizing with the
[PyTorch/XLA SPMD user guide][spmd-guide] first.

## Single device model + sharding configs

Going from single device training to distributed training usually doesn't require
changes to model code. For example, if we take a look at the [Llama][llama]
model, it doesn't call any sharding/parallelism APIs in the code. This makes for
a familiar experience for eager mode GPU users and is generally good software
engineering practice.

Instead, torchprime shards the model by modifying its parameters and layers
according to configurations specified at run time. The logic is implemented in
[`shard_model.py`][shard-model] and invoked from the [trainer][trainer]. Here is
an example sharding configuration that implements the [FSDP (aka ZeRO-3)][fsdp]
strategy for Llama dense models:

<!-- GitHub markdown embed -->
https://github.com/AI-Hypercomputer/torchprime/blob/b123c0cc157c28f32a0f6588f19e2d352d2a3617/torchprime/torch_xla_models/configs/model/sharding/llama-fsdp.yaml#L1-L17


> 📝 NOTE: Compared to the FSDP wrapper in PyTorch upstream that uses eager
> collective operations, torchprime stages out a computation graph corresponding
> to the training step where specific nodes in the graph are annotated with
> sharding constraints. The XLA compiler then propagates those constraints to all
> nodes in the graph and inserts the appropriate collective operations
> automatically. In contrast to eager PyTorch, the XLA compiler decides the best
> weight prefetching schedules.

## How to shard weights


## How to shard activations


### Indexing syntax


<!-- xrefs -->

[spmd-guide]: https://pytorch.org/xla/master/perf/spmd_basic.html
[llama]: ../torchprime/torch_xla_models/llama/model.py
[llama-fsdp]: ../torchprime/torch_xla_models/configs/model/sharding/llama-fsdp.yaml
[shard-model]: ../torchprime/sharding/shard_model.py
[trainer]: ../torchprime/torch_xla_models/train.py
[fsdp]: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
