#!/bin/bash
#
# Package the TensorFlow ROCm wheel built by scripts/run_tf_build_volume.sh into the
# Docker image that Triton's TensorFlow backend consumes (build.py passes it as
# TRITON_TENSORFLOW_DOCKER_IMAGE).
#
# Why this is needed: the backend's CMake (triton-inference-server-tensorflow_backend)
# creates a container from that image and `docker cp`s these files from a hard-coded path:
#   <BACKEND_TF_SITE>/tensorflow/libtensorflow_cc.so.2
#   <BACKEND_TF_SITE>/tensorflow/libtensorflow_framework.so.2
#   <BACKEND_TF_SITE>/tensorflow/include/LICENSE
# That path used to exist because the TF build ran inside `docker build`. With the
# host-persisted build (run_tf_build_volume.sh) the image has no TF, so this script
# unpacks the wheel's `tensorflow/` package to that path in a new image layer.
#
# Images:
#   ${TF_BUILD_IMAGE}  clean build environment (no TensorFlow); created by run_tf_build_volume.sh
#   ${TF_FINAL_IMAGE}  build env + packaged TensorFlow; the name build.py expects
#
# Usage (after a successful ./scripts/run_tf_build_volume.sh):
#   ./scripts/package_tf_wheel_image.sh

set -euo pipefail

TF_FINAL_IMAGE="${TF_FINAL_IMAGE:-localhost/debian12_rocm10.1.0_tensorflow}"
TF_BUILD_IMAGE="${TF_BUILD_IMAGE:-${TF_FINAL_IMAGE}_buildenv}"
TF_HOST_WORKSPACE="${TF_HOST_WORKSPACE:-$HOME/tf-workspace}"
TF_HOST_BAZEL_CACHE="${TF_HOST_BAZEL_CACHE:-$HOME/.cache/bazel-tf}"

# Must match TRITON_TENSORFLOW_PYTHON_PATH in the backend's CMakeLists.txt.
BACKEND_TF_SITE="${BACKEND_TF_SITE:-/root/.cache/bazel/_bazel_root/8f7dd4c2e00ecc9d40debccd14018321/external/python_3_12_x86_64-unknown-linux-gnu/lib/python3.12/site-packages}"

if ! docker image inspect "${TF_BUILD_IMAGE}" >/dev/null 2>&1; then
  echo "ERROR: build image ${TF_BUILD_IMAGE} not found."
  echo "       Run ./scripts/run_tf_build_volume.sh (without SKIP_IMAGE_BUILD=1) first."
  exit 1
fi

CTX="$(mktemp -d)"
trap 'rm -rf "${CTX}"' EXIT

echo "== Extracting the built wheel from the Bazel output =="
docker run --rm \
  -v "${TF_HOST_WORKSPACE}:/workspace" \
  -v "${TF_HOST_BAZEL_CACHE}:/root/.cache/bazel" \
  -v "${CTX}:/out" \
  "${TF_BUILD_IMAGE}" \
  bash -c 'set -e
    WHEEL="$(ls -1 /workspace/tensorflow-upstream/bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_rocm-*.whl | tail -n 1)"
    echo "Wheel: ${WHEEL}"
    cp -L "${WHEEL}" /out/tensorflow.whl'

cat > "${CTX}/Dockerfile" <<'EOF'
ARG TF_BUILD_IMAGE
FROM ${TF_BUILD_IMAGE}
ARG BACKEND_TF_SITE
COPY tensorflow.whl /tmp/tensorflow.whl
RUN mkdir -p "${BACKEND_TF_SITE}" \
 && unzip -q /tmp/tensorflow.whl 'tensorflow/*' -d "${BACKEND_TF_SITE}" \
 && rm /tmp/tensorflow.whl \
 && test -f "${BACKEND_TF_SITE}/tensorflow/libtensorflow_cc.so.2" \
 && test -f "${BACKEND_TF_SITE}/tensorflow/libtensorflow_framework.so.2" \
 && test -f "${BACKEND_TF_SITE}/tensorflow/include/LICENSE"
LABEL tf.packaged="true"
EOF

echo "== Building ${TF_FINAL_IMAGE} (build env + packaged TensorFlow) =="
docker build \
  --build-arg TF_BUILD_IMAGE="${TF_BUILD_IMAGE}" \
  --build-arg BACKEND_TF_SITE="${BACKEND_TF_SITE}" \
  -t "${TF_FINAL_IMAGE}" \
  "${CTX}"

echo "== Verifying files the TensorFlow backend copies =="
docker run --rm "${TF_FINAL_IMAGE}" ls -lL \
  "${BACKEND_TF_SITE}/tensorflow/libtensorflow_cc.so.2" \
  "${BACKEND_TF_SITE}/tensorflow/libtensorflow_framework.so.2" \
  "${BACKEND_TF_SITE}/tensorflow/include/LICENSE"
echo "Packaged TensorFlow into ${TF_FINAL_IMAGE}"
