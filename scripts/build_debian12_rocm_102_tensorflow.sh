#!/bin/bash
#
# Build Debian 12 + ROCm 10.2.0 TensorFlow repro image.
#
# Usage (from anywhere):
#   ./scripts/build_debian12_rocm_102_tensorflow.sh
#
# Skip the TensorFlow build/test that runs during `docker build`:
#   RUN_TF_REPRO_DURING_BUILD=0 ./scripts/build_debian12_rocm_102_tensorflow.sh
#
# ROCM_GFX_ARCHS selects the GPU architectures to install and to pass to
# TensorFlow as TF_ROCM_AMDGPU_TARGETS (comma-separated):
#   gfx942         default; MI300X / MI325X / MI300A
#   gfx942,gfx950  also MI355X
# A non-default set gets a tag suffix so images do not overwrite each other.
# build.py expects the default tag localhost/debian12_rocm10.2.0_tensorflow.

set -euo pipefail

DEFAULT_ARCHS="gfx942"
ROCM_GFX_ARCHS="${ROCM_GFX_ARCHS:-${DEFAULT_ARCHS}}"
DOCKERFILE="Dockerfile.debian12_rocm10.2.0_tensorflow"

if [ "${ROCM_GFX_ARCHS}" = "${DEFAULT_ARCHS}" ]; then
    IMAGE="localhost/debian12_rocm10.2.0_tensorflow"
else
    IMAGE="localhost/debian12_rocm10.2.0_tensorflow-${ROCM_GFX_ARCHS//,/_}"
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=============================================================="
echo "Building ${IMAGE} (ROCM_GFX_ARCHS=${ROCM_GFX_ARCHS})"
echo "=============================================================="
docker build \
  --build-arg "ROCM_GFX_ARCHS=${ROCM_GFX_ARCHS}" \
  --build-arg RUN_TF_REPRO_DURING_BUILD="${RUN_TF_REPRO_DURING_BUILD:-1}" \
  -t "${IMAGE}" \
  -f "${DOCKERFILE}" \
  .
echo ""

docker images | grep -E "${IMAGE}" || true
