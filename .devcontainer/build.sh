#!/usr/bin/env bash
set -e

# Path to the pre-built gdeflate install dropped into the repo folder.
# Place the gdeflate install tree at /workspace/gdeflate/ (i.e. copy
# libgdeflate/build/install/ → hipCOMP-core/gdeflate/).
GDEFLATE_ROOT="/workspace/gdeflate"

mkdir -p /workspace/build
cd /workspace/build

cmake /workspace \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -Dgdeflate_ROOT="${GDEFLATE_ROOT}"

make -j"$(nproc)" install

# Install the gdeflate runtime library so the system linker finds it.
cp "${GDEFLATE_ROOT}/lib/libgdeflate.so.1.0.0" /usr/local/lib/
ln -sf /usr/local/lib/libgdeflate.so.1.0.0 /usr/local/lib/libgdeflate.so.1
ln -sf /usr/local/lib/libgdeflate.so.1       /usr/local/lib/libgdeflate.so
ldconfig
