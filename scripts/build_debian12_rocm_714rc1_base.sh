#!/bin/bash
#
# Build Debian 12 + ROCm 7.14.0-rc1 base image

set -e

echo "========================================"
echo "Building localhost/debian12_rocm7.14.0-rc1"
echo "========================================"
docker build -t localhost/debian12_rocm7.14.0-rc1 -f Dockerfile.debian12_rocm7.14.0-rc1 .
echo ""

docker images | grep -E "localhost/debian12_rocm7.14.0-rc1" || true

