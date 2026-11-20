#!/bin/bash
#
# Build the Debian 12 + ROCm 10.2 base with CPython 3.12 on PATH.
#
# Builds localhost/debian12_rocm10.2.0 first (scripts/build_debian12_rocm_102_base.sh),
# then localhost/debian12_rocm10.2.0-py312 from Dockerfile.debian12_rocm10.2.0-py312.
# Set SKIP_BASE=1 to reuse an existing localhost/debian12_rocm10.2.0.
# Both images cover gfx942 (MI300X) and gfx950 (MI355X).

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
