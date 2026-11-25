#!/bin/bash
#
# Build Debian 12 + ROCm 10.2.0 (ROCm Core SDK) base image.
#
# By default the image installs ROCm for gfx942 (MI300X / MI308X / MI325X /
# MI300A) and gfx950 (MI355X), so the same image runs on MI300X and MI355X
# nodes and is tagged localhost/debian12_rocm10.2.0.
#
# For the MI300X-only image (build.py --mi300x-only) build the gfx942-only
# variant, tagged localhost/debian12_rocm10.2.0-gfx942:
#   ROCM_GFX_ARCHS=gfx942 ./scripts/build_debian12_rocm_102_base.sh

set -e

ROCM_GFX_ARCHS="${ROCM_GFX_ARCHS:-gfx942,gfx950}"
case "${ROCM_GFX_ARCHS}" in
    gfx942,gfx950) IMAGE_SUFFIX="" ;;
    gfx942)        IMAGE_SUFFIX="-gfx942" ;;
    *)
        echo "ROCM_GFX_ARCHS must be 'gfx942,gfx950' (default) or 'gfx942'; got '${ROCM_GFX_ARCHS}'" >&2
        exit 1
        ;;
esac

IMAGE_NAME="localhost/debian12_rocm10.2.0${IMAGE_SUFFIX}"
DOCKERFILE="Dockerfile.debian12_rocm10.2.0"

# The Dockerfile and build context live at the repo root.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "========================================"
echo "Building ${IMAGE_NAME} (${ROCM_GFX_ARCHS})"
echo "========================================"
# Empty build context: the Dockerfile does not COPY anything from the repo.
CONTEXT_DIR="$(mktemp -d)"
trap 'rm -rf "${CONTEXT_DIR}"' EXIT
docker build \
    --build-arg ROCM_GFX_ARCHS="${ROCM_GFX_ARCHS}" \
    -t "${IMAGE_NAME}" \
    -f "${DOCKERFILE}" \
    "${CONTEXT_DIR}"
echo ""

docker images | grep -E "${IMAGE_NAME}" || true
