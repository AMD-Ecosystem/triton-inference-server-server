#!/bin/bash
#
# Build Debian 12 + ROCm 7.2.3 base image

set -e

echo "========================================"
echo "Building localhost/debian12_rocm7.2.3"
echo "========================================"
docker build -t localhost/debian12_rocm7.2.3 -f Dockerfile.debian12_rocm7.2.3 .
echo ""

docker images | grep -E "localhost/debian12_rocm7.2.3" || true
