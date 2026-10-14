#!/bin/bash
#
# Build Debian 12 + ROCm 10.1.0 (ROCm Core SDK) base image

set -e

IMAGE_NAME="localhost/debian12_rocm10.1.0"
DOCKERFILE="Dockerfile.debian12_rocm10.1.0"

# The Dockerfile and build context live at the repo root.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "========================================"
echo "Building ${IMAGE_NAME}"
echo "========================================"
docker build -t "${IMAGE_NAME}" -f "${DOCKERFILE}" .
echo ""

docker images | grep -E "${IMAGE_NAME}" || true
