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
# The release image must run on both MI300X (gfx942, also MI308X / MI325X /
# MI300A) and MI355X (gfx950), so the Dockerfile's ROCM_GFX_ARCHS default
# (gfx942,gfx950) is always used.
# build.py expects the tag localhost/debian12_rocm10.2.0_tensorflow.

set -euo pipefail

DOCKERFILE="Dockerfile.debian12_rocm10.2.0_tensorflow"
IMAGE="localhost/debian12_rocm10.2.0_tensorflow"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=============================================================="
echo "Building ${IMAGE} (gfx942,gfx950)"
echo "=============================================================="
docker build \
  --build-arg RUN_TF_REPRO_DURING_BUILD="${RUN_TF_REPRO_DURING_BUILD:-1}" \
  -t "${IMAGE}" \
  -f "${DOCKERFILE}" \
  .
echo ""

docker images | grep -E "${IMAGE}" || true
