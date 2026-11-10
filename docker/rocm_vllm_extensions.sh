#!/bin/bash
# Install the ROCm 10.2 vLLM 0.32 stack from AMD prerelease wheels (Python 3.14),
# following the vLLM 0.32 ROCm 10.2 page: flash-attn, amd-aiter, triton, then the
# vLLM wheel. Then restore the ROCm torch 2.14 / torchvision 0.29.0 wheels, since
# the vLLM install can pull upstream torch over them.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="/opt/rocm/bin:${PATH}"
export ROCM_PATH=/opt/rocm
unset VLLM_USE_PRECOMPILED || true
unset VLLM_PRECOMPILED_WHEEL_LOCATION || true
unset VLLM_DOCKER_BUILD_CONTEXT || true

PREREL=https://rocm.frameworks-prereleases.amd.com/whl-multi-arch-staging/vllm
RC_INDEX=https://rc.repo.amd.com/rocm/whl-next/

apt-get update
apt-get install -y --no-install-recommends ca-certificates
rm -rf /var/lib/apt/lists/*

# --no-deps: these wheels depend on torch, and a normal install pulls the PyPI CUDA
# torch over the ROCm wheel. Their other dependencies are installed explicitly below.
python3.14 -m pip install --no-cache-dir --no-deps \
  "${PREREL}/flash-attn/flash_attn-2.8.3-cp314-cp314-linux_x86_64.whl" \
  "${PREREL}/amd-aiter/amd_aiter-0.1.24.post1-cp314-cp314-linux_x86_64.whl" \
  "${PREREL}/triton/triton-3.8.0%2Bgit669b31ac-cp314-cp314-linux_x86_64.whl"
python3.14 -m pip install --no-cache-dir \
  einops "pybind11>=3.0.1" ninja pandas psutil packaging "flydsl==0.3.4.1"

VLLM_WHL="${PREREL}/vllm/vllm-0.32.1.dev0%2Bg5df63a720.d20261031.rocm102-cp314-cp314-linux_x86_64.whl"
python3.14 -m pip install --no-cache-dir --no-deps "${VLLM_WHL}"

# The vLLM wheel requires amd-quark==0.12.post1, which is not on the public index.
# Install the other vLLM requirements from its metadata, skipping amd-quark, and pin
# the already-installed torch/torchvision/triton so pip cannot replace them.
python3.14 - << 'PY' > /tmp/vllm-reqs.txt
import importlib.metadata
from pip._vendor.packaging.requirements import Requirement

for line in importlib.metadata.requires("vllm") or []:
    req = Requirement(line)
    # amd-quark is not on the public index. flash-attn, amd-aiter and triton are already
    # installed from the explicit prerelease wheels above; letting pip resolve them here
    # falls back to the PyPI source build of flash-attn, which needs torch at build time.
    if req.name.lower() in ("amd-quark", "flash-attn", "amd-aiter", "triton", "vllm"):
        continue
    if req.marker is not None and not req.marker.evaluate({"extra": ""}):
        continue
    print(str(req))
PY
# PyJWT is installed by Debian without a RECORD file, so pip cannot uninstall it
# to upgrade. Install it over the top with --ignore-installed (no uninstall step).
python3.14 -m pip install --no-cache-dir --ignore-installed --no-deps PyJWT
python3.14 -m pip freeze | grep -iE '^(torch|torchvision|triton)==' > /tmp/vllm-constraints.txt || true
python3.14 -m pip install --no-cache-dir --extra-index-url "${RC_INDEX}" \
  -c /tmp/vllm-constraints.txt -r /tmp/vllm-reqs.txt

python3.14 -m pip install --no-cache-dir --force-reinstall --no-deps \
  --index-url "${RC_INDEX}" \
  "torch==2.14.0+rocm10.2.0rc1" \
  "torchvision==0.29.0+rocm10.2.0rc1"

python3.14 - << 'PY'
import importlib.metadata
import subprocess

names = []
for dist in importlib.metadata.distributions():
    name = dist.metadata["Name"]
    if name.lower().startswith(("nvidia-", "cuda-")):
        names.append(name)
if names:
    print("removing", " ".join(sorted(names)))
    subprocess.check_call(["python3.14", "-m", "pip", "uninstall", "-y", *names])
PY

ln -sfn /opt/rocm/share/amd_smi/amdsmi /usr/local/lib/python3.14/dist-packages/amdsmi
# The base image already has /etc/ld.so.conf.d/pytorch.conf pointing at the Python 3.12
# torch lib, so importing torch here would load the wrong libtorch_python.so. Find the
# 3.14 torch lib path without importing torch. The "00-" prefix sorts the file before
# pytorch.conf in any locale (locale collation ignores hyphens, so "pytorch-vllm.conf"
# would sort after "pytorch.conf").
TORCH_LIB=$(python3.14 -c 'import importlib.util,os; s=importlib.util.find_spec("torch"); print(os.path.join(os.path.dirname(s.origin),"lib"))')
rm -f /etc/ld.so.conf.d/pytorch-vllm.conf
echo "${TORCH_LIB}" > /etc/ld.so.conf.d/00-pytorch-vllm.conf
ldconfig
# The torch wheel bundles its own libamdhip64.so.7 (HIP 7.17.26434) under
# _rocm_sdk_core. Triton's server and libhipdelay load /opt/rocm's copy of the same
# HIP version. Two HIP runtimes in one process make the vLLM engine refuse to start
# ("refusing to mix ROCm installations"), and when both are loaded the process
# segfaults in hsa_shut_down at exit. Point the bundled path at /opt/rocm's copy so
# each process loads one runtime. The original is kept next to it.
SDK_LIB=$(python3.14 -c 'import importlib.util,os; s=importlib.util.find_spec("_rocm_sdk_core"); print(os.path.join(os.path.dirname(s.origin),"lib"))')
if [ -f /opt/rocm/lib/libamdhip64.so.7 ] && [ -f "${SDK_LIB}/libamdhip64.so.7" ] && [ ! -L "${SDK_LIB}/libamdhip64.so.7" ]; then
  mv "${SDK_LIB}/libamdhip64.so.7" "${SDK_LIB}/libamdhip64.so.7.therock-orig"
  ln -s /opt/rocm/lib/libamdhip64.so.7 "${SDK_LIB}/libamdhip64.so.7"
fi
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
python3.14 -c 'import amdsmi; amdsmi.amdsmi_init(); import torch, vllm; print(torch.__version__, torch.version.hip, vllm.__version__); amdsmi.amdsmi_shut_down()'
