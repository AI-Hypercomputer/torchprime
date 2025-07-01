# TPU vs. GPU: Accuracy Equivalence Despite the Precision Difference

This `README.md` compares the numerical results of training a ResNet-18 model on
TPU and GPU. 

## Understanding Numerical Differences: TPU vs. GPU

If you perform the exact same mathematical computation on different hardware
accelerators, will the result be identical? For deep learning workloads, the
answer is often no. Modern deep learning accelerators like Google's Tensor
Processing Units (TPUs) and NVIDIA's Graphics Processing Units (GPUs) employ
distinct floating-point precision levels and internal architectures to maximize
computational speed. This fundamental difference leads to the challenge of
testing for numerical equivalence in results, despite non-equivalent underlying
computations.

Consider a simple floating-point number, like `0.3`, when processed through
various deep learning hardware:
* A standard CPU operation, typically uses
  `float32`, which retains the full 23 bits of mantissa precision.
* An **NVIDIA GPU** (e.g., A100 and later) utilizes `TensorFloat-32` (TF32)
  within its Tensor Cores. While it accepts `float32` inputs, TF32 effectively
  processes them with an 8-bit exponent and a 10-bit mantissa internally. 
* A **Google TPU** leverages `bfloat16`. This 16-bit format is specifically
  designed with an 8-bit exponent (matching `float32`'s range) but reduces the
  mantissa (precision) to only 7 bits. This means `0.3` would be represented
  with a lower degree of precision than `float32`.


![alt text](img/bit_layout.svg "bit_layout")

Deep learning model training involves extensive calculations where small
numerical differences accumulate across hardware platforms. This makes direct,
bit-for-bit comparison of final model states (weights, gradients, or loss)
between different systems, like TPUs and GPUs, impractical and potentially
misleading.

However, deep learning models are remarkably robust. Even with these
computational variances, models trained on both TPUs and GPUs can converge to
nearly equivalent final model accuracy. TPUs often leverage lower precision
numerical formats, specifically bfloat16, which contributes to their speed and
energy efficiency by allowing for more computations per unit of time and reduced
memory usage. This document will demonstrate how, with correct setup and
understanding, TPUs can deliver high performance without compromising the
qualitative outcomes of your deep learning tasks.
[The research](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
    showed that neural networks were able to train with less precision

## Experimental Setup

### Model and Dataset

To provide a clear and focused comparison, we use the well-established
**ResNet-18** model. For the dataset, we use a combination of **VGGFace**, a
large-scale collection of face images, and **FaceID 550**. This creates a robust
benchmark for evaluating face recognition performance. The data loading logic,
detailed in `data.py`, creates a deterministic 90/10 stratified split for
training and testing, ensuring that images for every identity are present in
both sets.

### Methodology

Our methodology is designed to provide a comprehensive comparison by evaluating
two key aspects: the impact of **training techniques** and the numerical
differences between **hardware platforms**.

#### Training Approaches

The `train.py` script contains the core logic for our experiments. It defines
two distinct training functions to illustrate the impact of different training
strategies:

-   **`train_simple`**: This function uses a basic setup with a standard SGD
    optimizer and a fixed learning rate. It is used to demonstrate how
    sensitive training can be to hyperparameters, like a high learning rate.

-   **`train_prod`**: This function implements a more robust, production-like
    training approach, incorporating the AdamW optimizer, a learning rate
    scheduler, and backbone freezing.

By comparing these two approaches, our analysis in `viz.ipynb` highlights the
significant impact that production-level training techniques have on model
performance and stability.

#### Hardware Comparison and Statistical Analysis

To ensure a fair and robust comparison between TPU and GPU platforms, we follow
a rigorous process:

1.  **Multiple Runs**: We recognize that a single training run can be misleading
    due to random factors like weight initialization and data shuffling.
    Therefore, we execute the training process multiple times on both TPU and
    GPU platforms. This provides a distribution of results for each hardware
    type, allowing for a more reliable comparison than one based on a single
    run.

2.  **Learning Rate Selection**: Model performance is highly dependent on the
    choice of hyperparameters, particularly the learning rate. A learning rate
    that is optimal for one hardware platform may cause training to diverge on
    another. As detailed in our analysis notebook (`viz.ipynb`), we carefully
    selected a learning rate that ensures stable convergence on both TPU and
    GPU.

3.  **Statistical Analysis (t-test)**: A t-test is a statistical tool used to
    determine if there is a significant difference between the average results
    of two groups. We use an independent two-sample t-test to compare the final
    model accuracies from our multiple TPU and GPU runs. This is crucial because
    a single training run can be misleading due to random noise. The t-test
    allows us to confidently conclude whether the observed performance
    difference is real or just a product of chance.



### Results and Analysis

A detailed breakdown of the results, including visualizations of the accuracy
distributions and the full t-test calculations, is available in our analysis
notebook:

- [View Results Analysis](viz.ipynb)

The raw data and logs used for this analysis can be found in the following files:
- [metrics.csv](./metrics.csv)
- [gpu-prod.log](./gpu-prod.log)
- [tpu-prod.log](./tpu-prod.log)
