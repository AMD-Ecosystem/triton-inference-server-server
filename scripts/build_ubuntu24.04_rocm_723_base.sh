#!/bin/bash
#
# Build Ubuntu 24.04 + ROCm 7.2.3 base image

set -e

echo "========================================"
echo "Building localhost/ubuntu24.04_rocm7.2.3"
echo "========================================"
docker build --progress=plain -t localhost/ubuntu24.04_rocm7.2.3 -f Dockerfile.ubuntu24.04_rocm7.2.3 .
echo ""

docker images | grep -E "localhost/ubuntu24.04_rocm7.2.3" || true
