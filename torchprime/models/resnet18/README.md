# TPU vs. GPU: Accuracy Equivalence Despite the Precision Difference

In this tutorial, you'll learn the basics of numerical precision on GPUs and
TPUs and demonstrate that a model trained on an Nvidia GPU achieves equivalent
accuracy to one trained on a Google TPU.

## Understanding Numerical Differences: TPU vs. GPU

If you perform the exact same mathematical computation on different hardware
accelerators, will the result be identical? For deep learning workloads, the
answer is often no. Modern deep learning accelerators like Google's Tensor
Processing Units (TPUs) and NVIDIA's Graphics Processing Units (GPUs) can
default to different floating-point precision levels, which can yield slightly
different results while maximizing computational speed.

Consider how even a simple computation can vary across hardware due to their
distinct floating-point precision levels:

* A standard CPU operation, or full-precision mode on a GPU, typically uses
  `float32`, which retains the full 23 bits of mantissa precision.

* A **Google TPU** leverages `bfloat16`. This 16-bit format is specifically
  designed with an 8-bit exponent (matching `float32`'s range) but reduces the
  mantissa (precision) to only 7 bits.

* An **NVIDIA GPU** (e.g., A100 and newer) can default to different precisions.
  While they support `bfloat16`, their Tensor Cores also utilize
  `TensorFloat-32` (TF32) by default for `float32` operations, which processes
  them with a 10-bit mantissa. The precision level ultimately used often depends
  on the specific code settings.

![alt text](img/bit_layout.svg "bit_layout")

This difference in representation is why floating-point computations can yield
slightly different results on different hardware. Deep learning model training
involves extensive calculations where these small numerical differences
accumulate across hardware platforms. This makes direct, bit-for-bit comparison
of final model states (weights, gradients, or loss) between different systems,
like TPUs and GPUs, impractical and potentially misleading.

However, deep learning models are remarkably robust. Even with these
computational variances, models trained on both TPUs and GPUs can converge to
nearly equivalent final model accuracy. For more details on floating-point
precision, refer to this
[article](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
and
[tutorial](https://github.com/pytorch/xla/blob/9c8ae9f9d79770a0f534e7eccf5b48c087d7513f/docs/source/tutorials/precision_tutorial.ipynb).

## Experimental Setup

### Model and Dataset

To provide a clear and focused comparison, we use the well-established
**ResNet-18** model. For the dataset, we use a **CelebA**. The data is split
into 90/10 for training and testing, ensuring that images for every identity are
present in both sets. Implementation details can be found in
[model.py](model.py) and [data.py](../../torch_xla_vision_models/data.py).



### Methodology

Our methodology is designed to provide a fair comparison of the numerical
differences between **hardware platforms** by using a realistic training
approach.

#### Training Approaches

To achieve stable and reliable results for the hardware comparison, our training
process incorporates several well-established best practices:

-   **Optimizer**: We use the AdamW optimizer, which is known for its robust
    performance across a wide range of tasks.
-   **Learning Rate Scheduling**: A cosine learning rate scheduler with a warmup
    phase is used to help the model converge smoothly and avoid early
    divergence.
-   **Backbone Freezing**: The pretrained backbone of the ResNet-18 model is
    initially frozen. This allows the newly added classification layers to adapt
    to the dataset before fine-tuning the entire model, which improves
    stability.

This production-like setup ensures that our comparison between TPU and GPU
performance is based on a solid and reproducible training methodology. Check
[train.py](../../torch_xla_vision_models/train.py) for more detail.

#### Hardware Comparison and Statistical Analysis

To ensure a fair and robust comparison between TPU and GPU platforms, we follow
a rigorous process:

1.  **Multiple Runs**: We recognize that a single training run can be misleading
    due to random factors like weight initialization and data shuffling.
    Therefore, we execute the training process multiple times on both TPU and
    GPU platforms. This provides a distribution of results for each hardware
    type, allowing for a more reliable comparison than one based on a single
    run.

2.  **Statistical Analysis (t-test)**: A t-test is a statistical toolå used to
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
