#!/usr/bin/env bash
#
# Reproduce TensorFlow ROCm build (py3.12 target) and run a GPU visibility test.
#

set -euo pipefail

TF_REPO="${TF_REPO:-https://github.com/ROCm/tensorflow-upstream.git}"
TF_BRANCH="${TF_BRANCH:-develop-upstream}"
TF_WORKDIR="${TF_WORKDIR:-/workspace}"
TF_PYTHON_VERSION="${TF_PYTHON_VERSION:-3.12}"
TF_ROCM_AMDGPU_TARGETS="${TF_ROCM_AMDGPU_TARGETS:-gfx908,gfx90a,gfx942,gfx950,gfx1030,gfx1100,gfx1101,gfx1102,gfx1200,gfx1201}"
TF_CLANG_PATH="${TF_CLANG_PATH:-/usr/lib/llvm-19/bin/clang}"
TF_PATCH_FILE="${TF_PATCH_FILE:-/workspace/patches/tensorflow_tritonapi_build_develop-upstream_adjusted.patch}"

echo "== TensorFlow ROCm reproduce script =="
echo "Repo:   ${TF_REPO}"
echo "Branch: ${TF_BRANCH}"
echo "Workdir:${TF_WORKDIR}"
echo "Py:     ${TF_PYTHON_VERSION}"
echo "GPU targets: ${TF_ROCM_AMDGPU_TARGETS}"
echo "Clang:  ${TF_CLANG_PATH}"
echo "Patch:  ${TF_PATCH_FILE}"

mkdir -p "${TF_WORKDIR}"
cd "${TF_WORKDIR}"

if [ -d tensorflow-upstream/.git ]; then
  echo "Using existing tensorflow-upstream checkout"
else
  echo "Cloning tensorflow-upstream"
  git clone "${TF_REPO}" tensorflow-upstream
fi

cd tensorflow-upstream
git fetch --all --tags
git checkout "${TF_BRANCH}"

if [ -f "${TF_PATCH_FILE}" ]; then
  echo "Applying patch: ${TF_PATCH_FILE}"
  if git apply --check "${TF_PATCH_FILE}"; then
    git apply "${TF_PATCH_FILE}"
  else
    echo "ERROR: Patch does not apply cleanly: ${TF_PATCH_FILE}"
    exit 1
  fi
else
  echo "WARNING: Patch file not found, continuing unpatched: ${TF_PATCH_FILE}"
fi

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements_lock_3_12.txt

# Avoid SIGPIPE/141 from 'yes' under pipefail once configure stops reading.
set +o pipefail
yes "" | TF_NEED_CLANG=0 TF_NEED_ROCM=1 ROCM_PATH=/opt/rocm PYTHON_BIN_PATH=/usr/bin/python3 ./configure
set -o pipefail

bazel build \
  --config=rocm \
  --repo_env=HERMETIC_PYTHON_VERSION="${TF_PYTHON_VERSION}" \
  --repo_env=WHEEL_NAME=tensorflow_rocm \
  --repo_env=TF_ROCM_AMDGPU_TARGETS="${TF_ROCM_AMDGPU_TARGETS}" \
  --action_env=TF_PYTHON_VERSION="${TF_PYTHON_VERSION}" \
  --action_env=CLANG_COMPILER_PATH="${TF_CLANG_PATH}" \
  --action_env=CC="${TF_CLANG_PATH}" \
  --action_env=CXX="${TF_CLANG_PATH}++" \
  --action_env=TF_ROCM_AMDGPU_TARGETS="${TF_ROCM_AMDGPU_TARGETS}" \
  //tensorflow/tools/pip_package:wheel \
  --verbose_failures

WHEEL_PATH="$(ls -1 bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_rocm-*.whl | tail -n 1)"
echo "Built wheel: ${WHEEL_PATH}"

OUTPUT_BASE="$(bazel info output_base)"
PY312="${OUTPUT_BASE}/external/python_3_12_x86_64-unknown-linux-gnu/bin/python3.12"

if [ ! -x "${PY312}" ]; then
  echo "ERROR: Hermetic Python 3.12 not found at ${PY312}"
  exit 1
fi

"${PY312}" -m pip install --upgrade "${WHEEL_PATH}"

cd "${TF_WORKDIR}"

"${PY312}" - <<'PY'
import tensorflow as tf

# Query the list of physical GPU devices
gpus = tf.config.list_physical_devices("GPU")

# Print the total count
print(f"Number of GPUs available: {len(gpus)}")

# Optionally, list their details
for i, gpu in enumerate(gpus):
    print(f"GPU {i}: {gpu}")
PY

echo "Repro completed successfully."
echo "Wheel location: ${TF_WORKDIR}/tensorflow-upstream/${WHEEL_PATH}"
