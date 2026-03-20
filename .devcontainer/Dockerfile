FROM rocm/dev-ubuntu-22.04:6.4-complete

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    vim \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

ENV ROCM_PATH=/opt/rocm
ENV PATH="${ROCM_PATH}/bin:${PATH}"

WORKDIR /workspace
