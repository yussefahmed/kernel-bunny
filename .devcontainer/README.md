# Development container

This container provides:

- CUDA Toolkit 12.4.1 (`nvcc`) on Ubuntu 22.04
- GCC/G++ with C++17 support, CMake, Ninja and GDB
- Python 3 and `uv` 0.12.10
- VS Code support for Python, C++, CMake and NVIDIA Nsight
- RTX 3060 targeting through CUDA compute capability 8.6

The library path intentionally prefers the host driver injected by NVIDIA CDI.
This prevents CUDA's forward-compatibility library from taking precedence on
GeForce hardware when the container toolkit is newer than the host driver.

## Open it

1. Install the VS Code **Dev Containers** extension.
2. Make sure the host NVIDIA driver and NVIDIA Container Toolkit are installed.
   The container uses NVIDIA's CDI device name (`nvidia.com/gpu=all`).
3. In VS Code, run **Dev Containers: Reopen in Container**.

To create a Python 3.12 project environment inside the container:

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```

Check the complete toolchain at any time:

```bash
bash .devcontainer/verify-toolchain.sh
```

If GPU detection fails on the host, verify Docker GPU support first:

```bash
docker run --rm --device=nvidia.com/gpu=all \
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

On this host, the older `--gpus all` injection path produces an NVML error,
while the CDI command above has been verified successfully. If the CDI device
name is ever missing after a driver update, inspect it with `nvidia-ctk cdi list`
and restart the `nvidia-cdi-refresh` service before reopening the container.

The RTX 3060 is an Ampere GPU with compute capability 8.6. `CUDAARCHS=86` and
`TORCH_CUDA_ARCH_LIST=8.6` make CMake CUDA and PyTorch extension builds target
the installed GPU rather than compiling unnecessary architectures.
