#!/bin/bash
#
# Optional Ubuntu 26.04 + ROCm 10.2.0 thin base (deps + HIP linker path).
# python3 build.py --enable-rocm already uses rocm/rocm-staging:ubuntu-26.04_10.2.0rc1-full.

set -e

echo "=========================================="
echo "Building localhost/ubuntu26.04_rocm10.2.0rc1"
echo "=========================================="
docker build --progress=plain -t localhost/ubuntu26.04_rocm10.2.0rc1 \
  -f Dockerfile.ubuntu26.04_rocm10.2.0 .
echo ""

docker images | grep -E "localhost/ubuntu26.04_rocm10.2.0rc1|rocm/rocm-staging:ubuntu-26.04_10.2.0rc1-full" || true
