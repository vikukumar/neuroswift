# {NeuroSwift} Architecture Deep-Dive

NeuroSwift is built from the ground up to solve the **Quadratic Complexity** of modern Transformers, making high-performance LLMs accessible on standard CPUs.

---

## 1. Parallel Associative Scan (Linear SSM)

Traditional Transformers use Self-Attention, which has $O(N^2)$ complexity. This makes them slow and memory-intensive for long sequences.
**NeuroSwift** utilizes a **Linear State-Space Model (SSM)** where token mixing is performed via a **Parallel Associative Scan**.

- **Mechanism**: $h_t = \bar{A}h_{t-1} + \bar{B}x_t$.
- **Optimization**: By treating the scan as an associative prefix sum, we can compute it in $O(\log N)$ depth instead of $O(N)$ sequential steps.
- **CPU Boost**: Our implementation uses vectorized Numpy/Torch operations that bypass the GIL and Python overhead.

## 2. Sparse Mixture-of-Experts (MoE)

To keep the parameter count high while maintaining low inference latency, NeuroSwift uses **Sparse MoE**.

- **Routing**: A learned gating network selects the top-2 experts for each token.
- **Efficiency**: Instead of running a large 1B parameter FFN, we run two 50M parameter experts, achieving $10\times$ faster inference for the same capacity.
- **Load Balancing**: An auxiliary loss prevents "expert collapse," ensuring all experts are trained equally.

## 3. Online Hebbian Plasticity

NeuroSwift is the first architecture to integrate **Online Plasticity** for short-term memory.

- **Fast Weights**: A subset of weights is updated during the forward pass based on Hebbian learning rules: "Neurons that fire together, wire together."
- **Retrieval-Augmentation**: This allows the model to "remember" a document it just read in the prompt, even without explicit fine-tuning.

## 4. SSD-Backed Mmap Streaming

Our `MmapDataset` is engineered for **Extreme Data Capacity**.

- **Memory Mapping**: Tokens are served directly from the disk's file system cache.
- **Throughput**: By pre-tokenizing and mapping the files, we achieve **Zero-Copy** data loading, perfect for training on multi-gigabyte datasets with minimal RAM.

---
> [!NOTE]
> All NeuroSwift layers are designed to be **Device-Agnostic**. If a CUDA device is detected, the scan automatically dispatches to high-performance kernels; on CPU, it uses our custom vectorized scan.
