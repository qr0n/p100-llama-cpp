#!/usr/bin/env bash
# Build the bundled, patched llama.cpp for the Tesla P100 (sm_60).
#
#   ./build.sh                 build into llama.cpp/build, for the GPUs in this machine
#   CUDA_ARCH=60;61 ./build.sh build for P100 (sm_60) AND P40/GTX 10xx (sm_61) in one binary
#   JOBS=8 ./build.sh          limit parallel compile jobs (default: all cores)
#
# Needs CUDA 12.x (CUDA 13 dropped Pascal) and cmake. CUDA 12.4's nvcc rejects
# gcc newer than 13, so g++-13 is used as the host compiler when it is installed.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
src=$here/llama.cpp
jobs=${JOBS:-$(nproc)}

die() { echo "build.sh: $*" >&2; exit 1; }

# Default to the installed GPUs' own architectures. A plain sm_60 build still RUNS
# on sm_61 (P40/GTX 10xx), but compiles __dp4a as four scalar multiplies and picks
# the GP100-only MMVQ table, so a P40 silently gets the slow path.
if [[ -z ${CUDA_ARCH:-} ]] && command -v nvidia-smi >/dev/null; then
    CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                | tr -d . | sort -u | paste -sd';' || true)
fi
arch=${CUDA_ARCH:-60}
echo "build.sh: CUDA_ARCH=$arch" >&2

command -v cmake >/dev/null || die "cmake not found"
command -v nvcc  >/dev/null || die "nvcc not found (install CUDA 12.x and put its bin/ on PATH)"

cuda_major=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
[[ $cuda_major == 12 ]] || die "CUDA $cuda_major found; Pascal needs CUDA 12.x (13 removed sm_60/sm_61)"

compilers=()
if command -v gcc-13 >/dev/null && command -v g++-13 >/dev/null; then
    compilers=(-DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13
               -DCMAKE_CUDA_HOST_COMPILER=g++-13)
else
    echo "build.sh: gcc-13 not found, using the default compiler (CUDA 12.4 needs gcc <= 13)" >&2
fi

cmake -S "$src" -B "$src/build" \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$arch" \
    -DCMAKE_BUILD_TYPE=Release \
    "${compilers[@]}"
cmake --build "$src/build" -j "$jobs"

echo
echo "Built: $src/build/bin/"
"$src/build/bin/llama-server" --list-devices || true
