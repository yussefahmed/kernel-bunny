#include <cuda_runtime.h>


__global__ void sgemm_naive(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || column >= n) {
        return;
    }

    float sum = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
        sum += a[row * k + inner] * b[inner * n + column];
    }
    c[row * n + column] = sum;
}


extern "C" void launch_sgemm(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream
) {
    const dim3 block(16, 16);
    const dim3 grid(
        (n + block.x - 1) / block.x,
        (m + block.y - 1) / block.y
    );
    sgemm_naive<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}
