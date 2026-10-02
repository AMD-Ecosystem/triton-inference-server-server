#!/bin/bash
# Install the ROCm vLLM wheel, keep the ROCm torch, drop CUDA wheels, and
# compile the native extensions against that torch for gfx942.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="/opt/rocm/bin:${PATH}"
export ROCM_PATH=/opt/rocm
export VLLM_TARGET_DEVICE=rocm
export PYTORCH_ROCM_ARCH=gfx942
export MAX_JOBS=16
export CMAKE_BUILD_TYPE=Release
export VLLM_VERSION_OVERRIDE=0.27.1.dev5
unset VLLM_USE_PRECOMPILED || true
unset VLLM_PRECOMPILED_WHEEL_LOCATION || true
unset VLLM_DOCKER_BUILD_CONTEXT || true

apt-get update
apt-get install -y --no-install-recommends cmake ninja-build git g++ ca-certificates
rm -rf /var/lib/apt/lists/*

python3.14 -m pip download --no-deps -d /tmp/vllm-wheel --pre \
  --extra-index-url https://rocm.frameworks.amd.com/whl-multi-arch/vllm/ \
  "vllm==0.27.1.dev5+rocm10.0.0.gf46a9dfe2.d20260826" \
  "flash-attn==2.8.3" \
  "amd-aiter==0.1.20.post1"
python3.14 -m pip install --no-cache-dir --no-deps /tmp/vllm-wheel/*.whl

python3.14 - << 'PY'
import importlib.metadata
import subprocess
import sys

skip = {"amd-quark"}
pkgs = []
for req in importlib.metadata.requires("vllm") or []:
    if "extra ==" in req:
        continue
    spec = req.split(";")[0].strip()
    base = (
        spec.split("[")[0]
        .split("=")[0]
        .split("<")[0]
        .split(">")[0]
        .strip()
        .lower()
        .replace("_", "-")
    )
    if base in skip:
        print("skip", spec)
        continue
    pkgs.append(spec)
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--pre",
        "--ignore-installed",
        "--extra-index-url",
        "https://rocm.frameworks.amd.com/whl-multi-arch/vllm/",
        "--extra-index-url",
        "https://stable.repo.amd.com/rocm/whl-next/",
        *pkgs,
    ]
)
PY

python3.14 -m pip install --no-cache-dir --force-reinstall --no-deps \
  --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "torch==2.13.0+rocm10.0.0" \
  "torchvision==0.28.0+rocm10.0.0" \
  "triton==3.8.0+git4cff872c.rocm10.0.0"

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

rm -rf /tmp/vllm-src
mkdir -p /tmp/vllm-src
git -C /tmp/vllm-src init
git -C /tmp/vllm-src remote add origin https://github.com/vllm-project/vllm.git
git -C /tmp/vllm-src fetch --depth 1 origin f46a9dfe2c5f57bebbd29556cbbb25eabd874226
git -C /tmp/vllm-src checkout --detach FETCH_HEAD
python3.14 -m pip install --no-cache-dir \
  "packaging>=24.2" \
  "setuptools>=77.0.3,<80" \
  "setuptools-scm>=8" \
  "setuptools-rust>=1.9.0"
(
  cd /tmp/vllm-src
  python3.14 setup.py build_ext --inplace
)
python3.14 - << 'PY'
import shutil
from pathlib import Path

dst = Path("/usr/local/lib/python3.14/dist-packages/vllm")
for name in (
    "_C.abi3.so",
    "_rocm_C.abi3.so",
    "_C_stable_libtorch.abi3.so",
    "_moe_C_stable_libtorch.abi3.so",
):
    src = Path("/tmp/vllm-src/vllm") / name
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    shutil.copy2(src, dst / name)
    print("installed", dst / name, src.stat().st_size)
PY
rm -rf /tmp/vllm-src /tmp/vllm-wheel
ln -sfn /opt/rocm/share/amd_smi/amdsmi /usr/local/lib/python3.14/dist-packages/amdsmi
python3.14 -c 'import torch,os; open("/etc/ld.so.conf.d/pytorch-vllm.conf","w").write(os.path.join(os.path.dirname(torch.__file__),"lib")+"\n")'
ldconfig
python3.14 -c 'import amdsmi; amdsmi.amdsmi_init(); import torch; print(torch.__version__, torch.version.hip); import vllm._rocm_C; print("rocm_C", vllm._rocm_C.__file__); amdsmi.amdsmi_shut_down()'
