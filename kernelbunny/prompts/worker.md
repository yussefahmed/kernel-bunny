You are proposing exactly one optimization to an existing CUDA SGEMM kernel.

The public operation contract is:

- Compute C = A x B for FP32 row-major matrices.
- A has shape M x K, B has shape K x N, and C has shape M x N.
- Support arbitrary positive M, N, and K, including dimensions not aligned to a tile.
- Target an NVIDIA RTX 3060 with CUDA architecture sm_86.
- Preserve this exact exported launcher signature:

```cpp
extern "C" void launch_sgemm(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream
);
```

The harness independently compiles, checks, and benchmarks the code on
representative SGEMM workloads. Exact correctness and benchmark inputs are
intentionally hidden. Do not assume fixed dimensions.

Make one materially meaningful optimization. Return only a JSON object with
exactly these two fields:

```json
{
  "worker_intent": "A short explanation of the one optimization attempted.",
  "cuda_source": "The complete contents of candidate.cu, not a patch."
}
```

The CUDA source must be complete and must not be wrapped in Markdown fences.
