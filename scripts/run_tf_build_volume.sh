#!/bin/bash
#
# Build the Debian 12 + ROCm 10.1.0 TensorFlow repro image, then run the
# TensorFlow build in a container with host-persisted state so retries are
# incremental instead of re-cloning and recompiling from scratch.
#
# Persisted on the host:
#   ${TF_HOST_WORKSPACE:-$HOME/tf-workspace}        -> /workspace
#       (tensorflow-upstream checkout; cloned once, reused afterwards)
#   ${TF_HOST_BAZEL_CACHE:-$HOME/.cache/bazel-tf}   -> /root/.cache/bazel
#       (Bazel output base, external repos and compiled objects)
#   ${TF_HOST_PIP_CACHE:-$HOME/.cache/pip-tf}       -> /root/.cache/pip
#       (pip wheel downloads; the requirements lock pulls several GB)
#
# The build script and patch are mounted from the repo, so edits take effect
# without rebuilding the image.
#
# Usage (run from anywhere; run inside tmux for long builds):
#   ./scripts/run_tf_build_volume.sh
#   SKIP_IMAGE_BUILD=1 ./scripts/run_tf_build_volume.sh   # reuse existing image

set -euo pipefail

IMAGE="localhost/debian12_rocm10.1.0_tensorflow"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_HOST_WORKSPACE="${TF_HOST_WORKSPACE:-$HOME/tf-workspace}"
TF_HOST_BAZEL_CACHE="${TF_HOST_BAZEL_CACHE:-$HOME/.cache/bazel-tf}"
TF_HOST_PIP_CACHE="${TF_HOST_PIP_CACHE:-$HOME/.cache/pip-tf}"

mkdir -p "${TF_HOST_WORKSPACE}" "${TF_HOST_BAZEL_CACHE}" "${TF_HOST_PIP_CACHE}"
cd "${REPO_ROOT}"

# ${IMAGE} is the name build.py passes to the TensorFlow backend; after the build it holds
# the packaged TensorFlow (scripts/package_tf_wheel_image.sh). The build itself runs in a
# separate clean build-environment image so repeated packaging never stacks layers.
BUILD_IMAGE="${TF_BUILD_IMAGE:-${IMAGE}_buildenv}"

if [ "${SKIP_IMAGE_BUILD:-0}" != "1" ]; then
  echo "== Building ${IMAGE} (TensorFlow step deferred to container run) =="
  RUN_TF_REPRO_DURING_BUILD=0 ./scripts/build_debian12_rocm_101_tensorflow.sh
  docker tag "${IMAGE}" "${BUILD_IMAGE}"
elif ! docker image inspect "${BUILD_IMAGE}" >/dev/null 2>&1; then
  if [ "$(docker image inspect -f '{{if .Config.Labels}}{{index .Config.Labels "tf.packaged"}}{{end}}' "${IMAGE}" 2>/dev/null)" = "true" ]; then
    echo "ERROR: ${IMAGE} already contains packaged TensorFlow and ${BUILD_IMAGE} is missing."
    echo "       Re-run without SKIP_IMAGE_BUILD=1 to rebuild the clean build image."
    exit 1
  fi
  docker tag "${IMAGE}" "${BUILD_IMAGE}"
fi

# pip (running as root in the container) disables its cache unless the cache
# directory is owned by root, so fix the top-level directory ownership.
docker run --rm -v "${TF_HOST_PIP_CACHE}:/pip" "${IMAGE}" chown root:root /pip

CONTAINER_NAME="${TF_CONTAINER_NAME:-tf-rocm-build}"
if [ -n "$(docker ps -q -f "name=^${CONTAINER_NAME}$")" ]; then
  echo "ERROR: container '${CONTAINER_NAME}' is already running."
  echo "       Follow it with:  docker logs -f ${CONTAINER_NAME}"
  echo "       Or stop it with: docker stop ${CONTAINER_NAME}"
  exit 1
fi

echo "== Running TensorFlow build (workspace: ${TF_HOST_WORKSPACE}, bazel cache: ${TF_HOST_BAZEL_CACHE}) =="
# --init: the build script is otherwise PID 1, which ignores SIGINT/SIGTERM, so
# Ctrl+C would leave the container running unattached.
docker run --rm --init \
  --name "${CONTAINER_NAME}" \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size=16g \
  -v "${TF_HOST_WORKSPACE}:/workspace" \
  -v "${TF_HOST_BAZEL_CACHE}:/root/.cache/bazel" \
  -v "${TF_HOST_PIP_CACHE}:/root/.cache/pip" \
  -v "${REPO_ROOT}/patches:/patches:ro" \
  -v "${REPO_ROOT}/scripts/repro_tensorflow_rocm_build_test.sh:/usr/local/bin/repro_tensorflow_rocm_build_test.sh:ro" \
  -e TF_PATCH_FILE=/patches/tensorflow_tritonapi_build_develop-upstream_adjusted.patch \
  "${BUILD_IMAGE}" \
  /usr/local/bin/repro_tensorflow_rocm_build_test.sh

# Make the built wheel available to build.py's TensorFlow backend.
TF_BUILD_IMAGE="${BUILD_IMAGE}" TF_FINAL_IMAGE="${IMAGE}" \
  TF_HOST_WORKSPACE="${TF_HOST_WORKSPACE}" TF_HOST_BAZEL_CACHE="${TF_HOST_BAZEL_CACHE}" \
  "${REPO_ROOT}/scripts/package_tf_wheel_image.sh"
