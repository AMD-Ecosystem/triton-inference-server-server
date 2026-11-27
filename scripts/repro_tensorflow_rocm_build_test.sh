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

# Patch is required: a missing patch would silently produce an unpatched build.
if [ ! -f "${TF_PATCH_FILE}" ]; then
  echo "ERROR: Patch file not found: ${TF_PATCH_FILE}"
  exit 1
fi

# Persistent checkouts may already have the patch applied from a previous run.
if git apply --reverse --check "${TF_PATCH_FILE}" 2>/dev/null; then
  echo "Patch already applied, skipping: ${TF_PATCH_FILE}"
elif git apply --check "${TF_PATCH_FILE}"; then
  echo "Applying patch: ${TF_PATCH_FILE}"
  git apply "${TF_PATCH_FILE}"
else
  echo "ERROR: Patch does not apply cleanly: ${TF_PATCH_FILE}"
  exit 1
fi

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements_lock_3_12.txt

# Avoid SIGPIPE/141 from 'yes' under pipefail once configure stops reading.
set +o pipefail
yes "" | TF_NEED_CLANG=0 TF_NEED_ROCM=1 ROCM_PATH=/opt/rocm PYTHON_BIN_PATH=/usr/bin/python3 ./configure
set -o pipefail

# Bazel's external repo name depends on the bzlmod setup (e.g. "python_3_12_..." or
# "rules_python++python+python_3_12_..."), so search for the interpreter instead of
# hard-coding one path. Prints nothing if Bazel hasn't fetched it yet.
find_hermetic_python() {
  local py_tag="${TF_PYTHON_VERSION//./_}"
  ls -1 "$(bazel info output_base)"/external/*python_${py_tag}_x86_64-unknown-linux-gnu/bin/python"${TF_PYTHON_VERSION}" 2>/dev/null | head -n 1 || true
}

# Older versions of this script pip-installed the wheel into Bazel's hermetic interpreter.
# Bazel's build tools run on that interpreter, and an installed `tensorflow` shadows the
# source tree, so the TF API extraction step fails with "cannot import name 'test' from
# partially initialized module tensorflow._api.v2.__internal__". Clean a reused cache.
HERMETIC_PY="$(find_hermetic_python)"
if [ -n "${HERMETIC_PY}" ] && (cd /tmp && "${HERMETIC_PY}" -c "import tensorflow" >/dev/null 2>&1); then
  echo "Removing packages pip-installed into Bazel's hermetic Python: ${HERMETIC_PY}"
  "${HERMETIC_PY}" -m pip freeze --all | cut -d= -f1 | cut -d' ' -f1 | grep -vx pip \
    | xargs -r "${HERMETIC_PY}" -m pip uninstall -y
fi

# ROCm build flags. Do NOT use `--config=rocm` alone: in this tree it only expands to
# `no_tfrt`, and the ROCm toolchain (`rocm_clang_hermetic` in third_party/xla/tensorflow.bazelrc)
# is never imported, so the build silently produces a CPU-only wheel.
# These flags mirror `rocm_clang_hermetic` (rocm_base is defined in the root .bazelrc).
# `no_tfrt` still applies via the `build --config=rocm` line that ./configure writes.
ROCM_BAZEL_FLAGS=(
  --config=rocm_base
  --@rules_ml_toolchain//common:enable_rocm=True
  --@rules_ml_toolchain//common:enable_cuda=False
  --@rules_ml_toolchain//common:enable_sycl=False
  --@rules_ml_toolchain//common:enable_hermetic_cc=True
  --@local_config_rocm//rocm:rocm_path_type=hermetic
  # rocm_base selects the sysroot's libstdc++ (GCC 8), which lacks <span> needed by
  # flatbuffers' host tools. Use libc++ like the CPU build (common:linux). Must come
  # after --config=rocm_base so it overrides it.
  --@rules_ml_toolchain//common:stdlib=libc++
  --@rules_ml_toolchain//common:static_libcxx=true
  # The hermetic clang driver warns about -rtlib=compiler-rt on C compiles, and -Werror
  # from the toolchain turns that into a failure (e.g. brotli). Demote just that warning.
  --copt=-Wno-error=unused-command-line-argument
  --host_copt=-Wno-error=unused-command-line-argument
)

bazel build \
  "${ROCM_BAZEL_FLAGS[@]}" \
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

PY312="$(find_hermetic_python)"

if [ -z "${PY312}" ] || [ ! -x "${PY312}" ]; then
  echo "ERROR: Hermetic Python ${TF_PYTHON_VERSION} not found under $(bazel info output_base)/external"
  exit 1
fi

# Install into a throwaway venv, never into Bazel's hermetic interpreter: Bazel's build
# tools (e.g. the TF API extractor) run on that interpreter, and an installed `tensorflow`
# package shadows the source tree and breaks later builds with circular-import errors.
TEST_VENV=/tmp/tf-test-venv
rm -rf "${TEST_VENV}"
"${PY312}" -m venv "${TEST_VENV}"
TEST_PY="${TEST_VENV}/bin/python"
"${TEST_PY}" -m pip install --upgrade pip
"${TEST_PY}" -m pip install --upgrade "${WHEEL_PATH}"

cd "${TF_WORKDIR}"

# The hermetic-ROCm build links against ROCm Core SDK system deps (e.g.
# librocm_sysdeps_drm.so.2) that live outside /opt/rocm/lib. Only set this for the test:
# changing LD_LIBRARY_PATH before `bazel build` changes action keys and forces a rebuild.
export LD_LIBRARY_PATH="/opt/rocm/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"

"${TEST_PY}" - <<'PY'
import sys
import tensorflow as tf

# Fail loudly on a CPU-only build instead of reporting success.
if not tf.sysconfig.get_build_info().get("is_rocm_build", False):
    print("ERROR: installed TensorFlow is not a ROCm build (is_rocm_build=False)")
    sys.exit(1)

# Query the list of physical GPU devices
gpus = tf.config.list_physical_devices("GPU")

# Print the total count
print(f"Number of GPUs available: {len(gpus)}")

if not gpus:
    print("ERROR: TensorFlow sees no GPUs")
    sys.exit(1)

# Optionally, list their details
for i, gpu in enumerate(gpus):
    print(f"GPU {i}: {gpu}")

# Smoke test: run a kernel on the GPU, not just enumerate it.
with tf.device("/GPU:0"):
    a = tf.random.normal([1024, 1024])
    b = tf.matmul(a, a)
print(f"GPU matmul ran on {b.device}")
PY

echo "Repro completed successfully."
echo "Wheel location: ${TF_WORKDIR}/tensorflow-upstream/${WHEEL_PATH}"
