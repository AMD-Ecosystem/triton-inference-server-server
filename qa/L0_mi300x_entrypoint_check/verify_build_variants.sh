#!/bin/bash
# Dry-runs build.py (no images are built) and checks that:
#   * TRITON_MODEL_LOAD_THREAD_COUNT=1 is set only when the image holds the
#     TensorFlow backend together with ONNX Runtime or PyTorch (see
#     docker/entrypoint.d/80-model-load-threads.sh); single-backend images keep
#     Triton's default model-load threading.
#   * a default build of each single backend still targets gfx942 + gfx950 on
#     the two-arch images and has no MI300X-only marker. These are the
#     per-backend images that run on MI300X, MI308X, MI325X, MI350X and MI355X.
#   * --mi300x-only switches to the gfx942-only images, sets
#     TRITON_REQUIRE_MI300X=1 and PYTORCH_ROCM_ARCH=gfx942.
#
# --dryrun writes the generated files to ./build in the repository root, which
# is git-ignored. An existing ./build is overwritten.
#
# Usage: ./verify_build_variants.sh

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}" || exit 1
LOGS="$(mktemp -d)"
trap 'rm -rf "${LOGS}"' EXIT
RC=0
B=build
COMMON=(--dryrun --enable-rocm --no-container-pull --enable-logging --enable-stats
        --enable-metrics --endpoint=grpc --endpoint=http --linux-distro=debian)
FILES=("${B}/Dockerfile" "${B}/Dockerfile.buildbase" "${B}/Dockerfile.cibase" "${B}/cmake_build")

check() { # <label> <exit-status> <detail>
    if [ "$2" -eq 0 ]; then echo "  ok   $1"; else echo "  FAIL $1 ($3)"; RC=1; fi
}

for be in onnxruntime python pytorch tensorflow; do
    echo "== per-backend default build: ${be}"
    python3 build.py "${COMMON[@]}" --backend="${be}" > "${LOGS}/${be}.log" 2>&1
    check "build.py exits 0" $? "$(tail -n 2 "${LOGS}/${be}.log" | tr '\n' ' ')"
    grep -q "^ARG BASE_IMAGE=localhost/debian12_rocm10.2.0-py312$" "${B}/Dockerfile"
    check "base image is the two-arch localhost/debian12_rocm10.2.0-py312" $? "$(grep -m1 '^ARG BASE_IMAGE' "${B}/Dockerfile")"
    ! grep -q "TRITON_MODEL_LOAD_THREAD_COUNT" "${B}/Dockerfile"
    check "no TRITON_MODEL_LOAD_THREAD_COUNT in a single-backend image" $? "found variable"
    ! grep -qE "TRITON_REQUIRE_MI300X|com\.amd\.build\.gpu|debian12_rocm10\.2\.0[A-Za-z0-9_.-]*-gfx942" "${FILES[@]}" 2>/dev/null
    check "no MI300X-only marker or -gfx942 image name" $? "found marker"
    if [ "${be}" = "pytorch" ]; then
        grep -q "export PYTORCH_ROCM_ARCH=gfx942,gfx950" "${B}/cmake_build"
        check "PYTORCH_ROCM_ARCH=gfx942,gfx950" $? "$(grep -m1 'PYTORCH_ROCM_ARCH' "${B}/cmake_build")"
        grep -q "device-gfx942,device-gfx950" "${B}/Dockerfile.buildbase"
        check "PyTorch wheel has device-gfx942 and device-gfx950" $? "$(grep -m1 'device-gfx' "${B}/Dockerfile.buildbase")"
    fi
    if [ "${be}" = "tensorflow" ]; then
        grep -q "TRITON_TENSORFLOW_DOCKER_IMAGE=localhost/debian12_rocm10.2.0_tensorflow\b" "${B}/cmake_build" \
          && ! grep -q "TRITON_TENSORFLOW_DOCKER_IMAGE=[^ ]*-gfx942" "${B}/cmake_build"
        check "TensorFlow image is the two-arch localhost/debian12_rocm10.2.0_tensorflow" $? \
            "$(grep -o 'TRITON_TENSORFLOW_DOCKER_IMAGE=[^ "]*' "${B}/cmake_build" | head -n 1)"
    fi
done

echo "== all-in-one default (gfx942 + gfx950)"
python3 build.py "${COMMON[@]}" \
    --backend=onnxruntime --backend=python --backend=pytorch --backend=tensorflow \
    > "${LOGS}/allinone_default.log" 2>&1
check "build.py exits 0" $? "$(tail -n 2 "${LOGS}/allinone_default.log" | tr '\n' ' ')"
grep -q "ENV TRITON_MODEL_LOAD_THREAD_COUNT=1" "${B}/Dockerfile"
check "TRITON_MODEL_LOAD_THREAD_COUNT=1 set" $? ""
! grep -qE "TRITON_REQUIRE_MI300X|com\.amd\.build\.gpu" "${B}/Dockerfile"
check "no MI300X-only marker" $? "found marker"

echo "== TensorFlow with a CPU-only companion backend"
python3 build.py "${COMMON[@]}" --backend=python --backend=tensorflow > "${LOGS}/tf_python.log" 2>&1
check "build.py exits 0" $? "$(tail -n 2 "${LOGS}/tf_python.log" | tr '\n' ' ')"
! grep -q "TRITON_MODEL_LOAD_THREAD_COUNT" "${B}/Dockerfile"
check "no TRITON_MODEL_LOAD_THREAD_COUNT without a second GPU backend" $? "found variable"

echo "== all-in-one --mi300x-only"
python3 build.py "${COMMON[@]}" --mi300x-only \
    --backend=onnxruntime --backend=python --backend=pytorch --backend=tensorflow \
    > "${LOGS}/allinone.log" 2>&1
check "build.py exits 0" $? "$(tail -n 2 "${LOGS}/allinone.log" | tr '\n' ' ')"
grep -q "^ARG BASE_IMAGE=localhost/debian12_rocm10.2.0-py312-gfx942$" "${B}/Dockerfile"
check "base image is gfx942-only" $? "$(grep -m1 '^ARG BASE_IMAGE' "${B}/Dockerfile")"
grep -q "ENV TRITON_REQUIRE_MI300X=1" "${B}/Dockerfile"
check "TRITON_REQUIRE_MI300X=1 set" $? ""
grep -q "ENV TRITON_MODEL_LOAD_THREAD_COUNT=1" "${B}/Dockerfile"
check "TRITON_MODEL_LOAD_THREAD_COUNT=1 set" $? ""
grep -q "export PYTORCH_ROCM_ARCH=gfx942$" "${B}/cmake_build"
check "PYTORCH_ROCM_ARCH=gfx942" $? ""
grep -q "TRITON_TENSORFLOW_DOCKER_IMAGE=localhost/debian12_rocm10.2.0_tensorflow-gfx942" "${B}/cmake_build"
check "TensorFlow image is gfx942-only" $? ""

echo "== --mi300x-only rejects other settings"
python3 build.py --dryrun --mi300x-only > "${LOGS}/bad1.log" 2>&1
[ $? -ne 0 ] && grep -q "requires --enable-rocm" "${LOGS}/bad1.log"
check "--mi300x-only without --enable-rocm is rejected" $? ""
python3 build.py --dryrun --enable-rocm --mi300x-only --rocm-arch gfx950 > "${LOGS}/bad2.log" 2>&1
[ $? -ne 0 ] && grep -q "gfx942 only" "${LOGS}/bad2.log"
check "--mi300x-only with --rocm-arch gfx950 is rejected" $? ""

[ "${RC}" -eq 0 ] && echo "*** Test Passed" || echo "*** Test FAILED"
exit "${RC}"
