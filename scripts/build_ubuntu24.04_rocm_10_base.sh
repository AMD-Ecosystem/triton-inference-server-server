#!/bin/bash
#
# Optional Ubuntu 24.04 + ROCm 10.0.0 thin base (deps + HIP linker path).
# python3 build.py --enable-rocm already uses rocm/dev-ubuntu-24.04:10.0.0-full.

set -e

echo "=========================================="
echo "Building localhost/ubuntu24.04_rocm10.0.0"
echo "=========================================="
docker build --progress=plain -t localhost/ubuntu24.04_rocm10.0.0 \
  -f Dockerfile.ubuntu24.04_rocm10.0.0 .
echo ""

docker images | grep -E "localhost/ubuntu24.04_rocm10.0.0|rocm/dev-ubuntu-24.04:10.0.0-full" || true
