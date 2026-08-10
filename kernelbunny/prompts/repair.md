You are repairing one failed CUDA SGEMM candidate. Preserve its intended
optimization when practical, but correctness is more important than preserving
a broken technique.

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

The harness independently controls repair limits, compilation, correctness,
and benchmarking. Do not assume fixed or disclosed test dimensions.

Return only a JSON object with exactly these fields:

```json
{
  "worker_intent": "A short explanation of the defect repaired.",
  "cuda_source": "The complete repaired candidate.cu, not a patch."
}
```

The CUDA source must be complete and must not be wrapped in Markdown fences.
