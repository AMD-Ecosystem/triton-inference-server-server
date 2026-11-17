#!/bin/bash
#
# Build the Debian 12 + ROCm 10.2 base with CPython 3.12 on PATH.
#
# Builds localhost/debian12_rocm10.2.0 first (scripts/build_debian12_rocm_102_base.sh),
# then localhost/debian12_rocm10.2.0-py312 from Dockerfile.debian12_rocm10.2.0-py312.
# Set SKIP_BASE=1 to reuse an existing localhost/debian12_rocm10.2.0.
#
# Only the default ROCM_GFX_ARCHS (gfx942,gfx950) is supported here, because the
# py312 layer is built on the default base tag.

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ "${SKIP_BASE:-0}" != "1" ]; then
    ./scripts/build_debian12_rocm_102_base.sh
fi

CONTEXT_DIR="$(mktemp -d)"
trap 'rm -rf "${CONTEXT_DIR}"' EXIT

echo "========================================"
echo "Building localhost/debian12_rocm10.2.0-py312"
echo "========================================"
docker build \
    -t localhost/debian12_rocm10.2.0-py312 \
    -f Dockerfile.debian12_rocm10.2.0-py312 \
    "${CONTEXT_DIR}"

docker images | grep -E "localhost/debian12_rocm10.2.0(-py312)?" || true
