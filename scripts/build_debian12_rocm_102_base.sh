#!/bin/bash
#
# Build Debian 12 + ROCm 10.2.0 (ROCm Core SDK) base image.
#
# ROCM_GFX_ARCHS selects the GPU architectures to install (comma-separated):
#   gfx942,gfx950  default; runs on MI300X/MI325X and MI355X hosts
#   gfx942         MI300X-only image
#   gfx950         MI355X-only image
# A non-default set gets a tag suffix so images for different sets do not overwrite
# each other, for example localhost/debian12_rocm10.2.0-gfx950.

set -e

DEFAULT_ARCHS="gfx942,gfx950"
ROCM_GFX_ARCHS="${ROCM_GFX_ARCHS:-${DEFAULT_ARCHS}}"
DOCKERFILE="Dockerfile.debian12_rocm10.2.0"

if [ "${ROCM_GFX_ARCHS}" = "${DEFAULT_ARCHS}" ]; then
    IMAGE_NAME="localhost/debian12_rocm10.2.0"
else
    IMAGE_NAME="localhost/debian12_rocm10.2.0-${ROCM_GFX_ARCHS//,/_}"
fi

# The Dockerfile and build context live at the repo root.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "========================================"
echo "Building ${IMAGE_NAME} (ROCM_GFX_ARCHS=${ROCM_GFX_ARCHS})"
echo "========================================"
# Empty build context: the Dockerfile does not COPY anything from the repo.
CONTEXT_DIR="$(mktemp -d)"
trap 'rm -rf "${CONTEXT_DIR}"' EXIT
docker build \
    --build-arg "ROCM_GFX_ARCHS=${ROCM_GFX_ARCHS}" \
    -t "${IMAGE_NAME}" \
    -f "${DOCKERFILE}" \
    "${CONTEXT_DIR}"
echo ""

docker images | grep -E "${IMAGE_NAME}" || true
