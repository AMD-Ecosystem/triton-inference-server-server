#!/bin/bash
#
# Build Debian 12 + ROCm 7.14.0-rc1 TensorFlow repro image
#

set -euo pipefail

echo "=============================================================="
echo "Building localhost/debian12_rocm7.14.0-rc1_tensorflow"
echo "=============================================================="
docker build \
  -t localhost/debian12_rocm7.14.0-rc1_tensorflow \
  -f Dockerfile.debian12_rocm7.14.0-rc1_tensorflow \
  .
echo ""

docker images | grep -E "localhost/debian12_rocm7.14.0-rc1_tensorflow" || true
