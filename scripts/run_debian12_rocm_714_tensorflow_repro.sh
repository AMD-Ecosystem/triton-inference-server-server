#!/bin/bash
#
# Run TensorFlow ROCm build+test repro inside the dedicated image.
#

set -euo pipefail

IMAGE="${IMAGE:-localhost/debian12_rocm7.14.0_tensorflow}"
HOST_DIR="${HOST_DIR:-/home/jparada/072026/triton-inference-server-server}"

echo "=============================================================="
echo "Running TensorFlow ROCm repro in ${IMAGE}"
echo "Host mount: ${HOST_DIR} -> /workspace"
echo "=============================================================="

docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size=16g \
  -v "${HOST_DIR}:/workspace" \
  "${IMAGE}" \
  /usr/local/bin/repro_tensorflow_rocm_build_test.sh
