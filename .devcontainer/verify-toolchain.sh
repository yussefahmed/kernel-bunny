#!/usr/bin/env bash
set -u

echo "Toolchain"
uv --version
python3 --version
g++ --version | head -n 1
cmake --version | head -n 1
nvcc --version | tail -n 1

echo
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader; then
    echo "CUDA GPU access is ready."
else
    echo "CUDA toolkit is installed, but the GPU is not visible."
    echo "Install/configure NVIDIA Container Toolkit on the host, then rebuild the container."
fi

