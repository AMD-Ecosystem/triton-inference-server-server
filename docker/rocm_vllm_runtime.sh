#!/bin/bash
# Delay libamdhip64 until after the ROCm Triton wheel loads, and make the
# vLLM backend stub use Python 3.14 and amdsmi before torch.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends gcc libc6-dev patchelf
rm -rf /var/lib/apt/lists/*

cat > /tmp/hipdelay.c << 'EOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static void *
real_hip(void)
{
  static void *handle;
  if (handle == NULL) {
    handle = dlopen("/opt/rocm/lib/libamdhip64.so.7", RTLD_NOW | RTLD_GLOBAL);
    if (handle == NULL) {
      fprintf(stderr, "libhipdelay: %s\n", dlerror());
      abort();
    }
  }
  return handle;
}

#define FWD(ret, name, decl, call)                              \
  ret name decl                                                \
  {                                                             \
    static ret (*fn) decl;                                      \
    if (fn == NULL) {                                           \
      fn = (ret(*) decl)dlsym(real_hip(), #name);              \
      if (fn == NULL) {                                         \
        fprintf(stderr, "libhipdelay: %s\n", dlerror());        \
        abort();                                                \
      }                                                         \
    }                                                           \
    return fn call;                                             \
  }

FWD(hipError_t, hipStreamSynchronize, (hipStream_t stream), (stream))
FWD(hipError_t, hipSetDevice, (int device), (device))
FWD(hipError_t, hipStreamCreate, (hipStream_t * stream), (stream))
FWD(hipError_t, hipMemcpyPeer, (void *dst, int dstDevice, const void *src, int srcDevice, size_t size), (dst, dstDevice, src, srcDevice, size))
FWD(hipError_t, hipIpcOpenMemHandle, (void **ptr, hipIpcMemHandle_t handle, unsigned int flags), (ptr, handle, flags))
FWD(hipError_t, hipStreamDestroy, (hipStream_t stream), (stream))
FWD(const char *, hipGetErrorString, (hipError_t err), (err))
FWD(hipError_t, hipMemcpy, (void *dst, const void *src, size_t size, hipMemcpyKind kind), (dst, src, size, kind))
FWD(hipError_t, hipGetDevice, (int *device), (device))
FWD(hipError_t, hipIpcGetMemHandle, (hipIpcMemHandle_t * handle, void *ptr), (handle, ptr))
FWD(hipError_t, hipIpcCloseMemHandle, (void *ptr), (ptr))
EOF

cat > /tmp/hipdelay.map << 'EOF'
hip_4.2 {
  global:
    hipStreamSynchronize;
    hipSetDevice;
    hipStreamCreate;
    hipMemcpyPeer;
    hipIpcOpenMemHandle;
    hipStreamDestroy;
    hipGetErrorString;
    hipMemcpy;
    hipGetDevice;
    hipIpcGetMemHandle;
    hipIpcCloseMemHandle;
  local:
    *;
};
EOF

gcc -shared -fPIC -O2 -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include \
  /tmp/hipdelay.c -ldl -Wl,--version-script=/tmp/hipdelay.map \
  -o /opt/tritonserver/backends/python/libhipdelay.so
stub=/opt/tritonserver/backends/python/triton_python_backend_stub
if [ -f "${stub}.bin" ]; then
  stub="${stub}.bin"
fi
patchelf --replace-needed libamdhip64.so.7 libhipdelay.so "$stub"
patchelf --set-rpath '$ORIGIN:/opt/rocm/lib' "$stub"
if [ "$stub" != /opt/tritonserver/backends/python/triton_python_backend_stub ]; then
  mv "$stub" /opt/tritonserver/backends/python/triton_python_backend_stub
fi
chmod +x /opt/tritonserver/backends/python/triton_python_backend_stub

python3.14 - << 'PY'
from pathlib import Path

path = Path("/opt/tritonserver/backends/vllm/model.py")
text = path.read_text()
marker = 'sys.executable = "/usr/bin/python3.14"'
if marker not in text:
    path.write_text(
        "import sys\n"
        'sys.executable = "/usr/bin/python3.14"\n'
        "import amdsmi as _amdsmi\n"
        "_amdsmi.amdsmi_init()\n"
        "_amdsmi.amdsmi_get_processor_handles()\n"
        "_amdsmi.amdsmi_shut_down()\n"
        + text
    )
PY

python3.14 -m pip install --no-cache-dir --upgrade "openai>=2.0.0"
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
rm -f /tmp/hipdelay.c /tmp/hipdelay.map
# L0 clients invoke "python3". The server wheel stays on /usr/bin/python3
# (3.12); vLLM and its client imports live on 3.14.
ln -sfn /usr/bin/python3.14 /usr/local/bin/python3
ln -sfn /usr/bin/python3.14 /usr/local/bin/python
# Every Python 3.14 process, including the accuracy-test client, has to
# initialize amdsmi before torch or vLLM stays on an empty device string.
# /usr/lib/python3.14/sitecustomize.py is the file Python actually loads;
# a copy under dist-packages is ignored while this one exists.
cat > /usr/lib/python3.14/sitecustomize.py << 'EOF'
import amdsmi as _amdsmi

_amdsmi.amdsmi_init()
_amdsmi.amdsmi_get_processor_handles()
_amdsmi.amdsmi_shut_down()
EOF
