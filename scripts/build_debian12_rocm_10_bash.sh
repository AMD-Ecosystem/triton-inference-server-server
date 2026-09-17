#!/bin/bash
#
# Build Debian 12 + ROCm 10.0.0 (ROCm Core SDK) base image

set -e

echo "========================================"
echo "Building localhost/debian12_rocm10.0.0"
echo "========================================"
docker build -t localhost/debian12_rocm10.0.0 -f Dockerfile.debian12_rocm10.0.0 .
echo ""

docker images | grep -E "localhost/debian12_rocm10.0.0" || true
