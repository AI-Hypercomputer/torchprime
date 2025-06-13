# DLRM

Currently implemented Dlrm v1 & v2 models using dense Embeddings and torchax training pipeline.
For demonstration and experimentation purposes, embeddings are using Jax based Flaxx NN module and integrated into a pure pytorch moodel code.
This is supposed to be replaced by more efficient implementations using SparceCores when they are available.

One can run "fully sized" dlrm v2 dataset using this implementation and it is scaled in the original paper way - model parallel for embeddings, then commnucation call to compute all-to-all features interactions and then data parallel execution for the rest of the layers.
Tested on v6-8 slice and with sgd optimizer on synthetic data. 

To run the training on synthetic random data:

```sh
cd {torchprime root dir}
python torchprime/experimental/torchax_models/train_rec.py
```


 