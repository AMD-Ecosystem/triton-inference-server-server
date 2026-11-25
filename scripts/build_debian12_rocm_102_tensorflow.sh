#!/bin/bash
#
# Build Debian 12 + ROCm 10.2.0 TensorFlow repro image.
#
# Usage (from anywhere):
#   ./scripts/build_debian12_rocm_102_tensorflow.sh
#
# Skip the TensorFlow build/test that runs during `docker build`:
#   RUN_TF_REPRO_DURING_BUILD=0 ./scripts/build_debian12_rocm_102_tensorflow.sh
#
# By default the image runs on both MI300X (gfx942, also MI308X / MI325X /
# MI300A) and MI355X (gfx950): the Dockerfile's ROCM_GFX_ARCHS default
# (gfx942,gfx950) is used and build.py expects the tag
# localhost/debian12_rocm10.2.0_tensorflow.
#
# For the MI300X-only image (build.py --mi300x-only) build for gfx942 only,
# tagged localhost/debian12_rocm10.2.0_tensorflow-gfx942:
#   ROCM_GFX_ARCHS=gfx942 ./scripts/build_debian12_rocm_102_tensorflow.sh

set -euo pipefail

ROCM_GFX_ARCHS="${ROCM_GFX_ARCHS:-gfx942,gfx950}"
case "${ROCM_GFX_ARCHS}" in
    gfx942,gfx950) IMAGE_SUFFIX="" ;;
    gfx942)        IMAGE_SUFFIX="-gfx942" ;;
    *)
        echo "ROCM_GFX_ARCHS must be 'gfx942,gfx950' (default) or 'gfx942'; got '${ROCM_GFX_ARCHS}'" >&2
        exit 1
        ;;
esac

DOCKERFILE="Dockerfile.debian12_rocm10.2.0_tensorflow"
IMAGE="localhost/debian12_rocm10.2.0_tensorflow${IMAGE_SUFFIX}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=============================================================="
echo "Building ${IMAGE} (${ROCM_GFX_ARCHS})"
echo "=============================================================="
docker build \
  --build-arg ROCM_GFX_ARCHS="${ROCM_GFX_ARCHS}" \
  --build-arg RUN_TF_REPRO_DURING_BUILD="${RUN_TF_REPRO_DURING_BUILD:-1}" \
  -t "${IMAGE}" \
  -f "${DOCKERFILE}" \
  .
echo ""

docker images | grep -E "${IMAGE}" || true
