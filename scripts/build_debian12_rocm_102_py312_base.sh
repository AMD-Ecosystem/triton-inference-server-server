#!/bin/bash
#
# Build the Debian 12 + ROCm 10.2 base with CPython 3.12 on PATH.
#
# Builds localhost/debian12_rocm10.2.0 first (scripts/build_debian12_rocm_102_base.sh),
# then localhost/debian12_rocm10.2.0-py312 from Dockerfile.debian12_rocm10.2.0-py312.
# Set SKIP_BASE=1 to reuse an existing localhost/debian12_rocm10.2.0.
# By default both images cover gfx942 (MI300X) and gfx950 (MI355X).
#
# For the MI300X-only image (build.py --mi300x-only) build the gfx942-only
# variants, tagged localhost/debian12_rocm10.2.0-gfx942 and
# localhost/debian12_rocm10.2.0-py312-gfx942:
#   ROCM_GFX_ARCHS=gfx942 ./scripts/build_debian12_rocm_102_py312_base.sh

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROCM_GFX_ARCHS="${ROCM_GFX_ARCHS:-gfx942,gfx950}"
case "${ROCM_GFX_ARCHS}" in
    gfx942,gfx950) IMAGE_SUFFIX="" ;;
    gfx942)        IMAGE_SUFFIX="-gfx942" ;;
    *)
        echo "ROCM_GFX_ARCHS must be 'gfx942,gfx950' (default) or 'gfx942'; got '${ROCM_GFX_ARCHS}'" >&2
        exit 1
        ;;
esac
export ROCM_GFX_ARCHS

PARENT_IMAGE="localhost/debian12_rocm10.2.0${IMAGE_SUFFIX}"
IMAGE_NAME="localhost/debian12_rocm10.2.0-py312${IMAGE_SUFFIX}"

if [ "${SKIP_BASE:-0}" != "1" ]; then
    ./scripts/build_debian12_rocm_102_base.sh
fi

CONTEXT_DIR="$(mktemp -d)"
trap 'rm -rf "${CONTEXT_DIR}"' EXIT

echo "========================================"
echo "Building ${IMAGE_NAME} (${ROCM_GFX_ARCHS})"
echo "========================================"
docker build \
    --build-arg BASE_IMAGE="${PARENT_IMAGE}" \
    -t "${IMAGE_NAME}" \
    -f Dockerfile.debian12_rocm10.2.0-py312 \
    "${CONTEXT_DIR}"

docker images | grep -E "localhost/debian12_rocm10.2.0(-py312)?${IMAGE_SUFFIX}" || true
