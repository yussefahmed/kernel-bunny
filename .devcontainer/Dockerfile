FROM ghcr.io/astral-sh/uv:0.12.10 AS uv

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        gdb \
        git \
        ninja-build \
        openssh-client \
        pkg-config \
        python3 \
        python3-dev \
        python3-venv \
        sudo \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}" \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}"

ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    CUDAARCHS=86 \
    TORCH_CUDA_ARCH_LIST=8.6

RUN mkdir -p /opt/uv/python \
    && chown -R "${USERNAME}:${USERNAME}" /opt/uv

USER ${USERNAME}
WORKDIR /workspaces/KernelBunny

