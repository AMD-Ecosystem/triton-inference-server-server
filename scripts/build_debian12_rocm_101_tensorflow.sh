#!/bin/bash
#
# Build Debian 12 + ROCm 10.1.0 TensorFlow repro image
#
# Usage (run from the triton-inference-server-server directory):
#   ./scripts/build_debian12_rocm_101_tensorflow.sh
#
# Skip the TensorFlow build/test that runs during `docker build`:
#   RUN_TF_REPRO_DURING_BUILD=0 ./scripts/build_debian12_rocm_101_tensorflow.sh

set -euo pipefail

IMAGE="localhost/debian12_rocm10.1.0_tensorflow"
DOCKERFILE="Dockerfile.debian12_rocm10.1.0_tensorflow"

echo "=============================================================="
echo "Building ${IMAGE}"
echo "=============================================================="
docker build \
  --build-arg RUN_TF_REPRO_DURING_BUILD="${RUN_TF_REPRO_DURING_BUILD:-1}" \
  -t "${IMAGE}" \
  -f "${DOCKERFILE}" \
  .
echo ""

docker images | grep -E "${IMAGE}" || true
