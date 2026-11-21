#!/bin/bash
#
# Build Debian 12 + ROCm 10.2.0 (ROCm Core SDK) base image.
#
# The image installs ROCm for gfx942 (MI300X / MI308X / MI325X / MI300A) and
# gfx950 (MI355X), so the same images run on MI300X and MI355X nodes.

set -e

IMAGE_NAME="localhost/debian12_rocm10.2.0"
DOCKERFILE="Dockerfile.debian12_rocm10.2.0"

# The Dockerfile and build context live at the repo root.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "========================================"
echo "Building ${IMAGE_NAME} (gfx942,gfx950)"
echo "========================================"
# Empty build context: the Dockerfile does not COPY anything from the repo.
CONTEXT_DIR="$(mktemp -d)"
trap 'rm -rf "${CONTEXT_DIR}"' EXIT
docker build \
    -t "${IMAGE_NAME}" \
    -f "${DOCKERFILE}" \
    "${CONTEXT_DIR}"
echo ""

docker images | grep -E "${IMAGE_NAME}" || true
