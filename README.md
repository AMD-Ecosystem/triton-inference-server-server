<!--
# Copyright 2018-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->
[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)


# Triton Inference Server - ROCm Edition

----

## ROCm Support

This repository contains ROCm-enabled builds of Triton Inference Server for AMD GPUs. The following backends are available or in development:

- **ONNX Runtime Backend** - Done
- **Python Backend** - Done
- **vLLM Backend** - Done
- **PyTorch Backend**  - Done
- **Tensorflow Backend** - Done

### ROCm-Enabled Repository Branches

The following table lists the ROCm-enabled Triton Inference Server component repositories. Components are based on **Triton Inference Server r26.09** except tensorflow backend.

> **Note**: These repositories contain ROCm-enabled source code. For the remaining repositories used to build the Triton Server artifacts without need of ROCm enablement, we use them as-is from the Triton Inference Server GitHub repository https://github.com/triton-inference-server.

| Component | Repository | Branch |
|-----------|------------|--------|
| Server | [AMD-Ecosystem/triton-inference-server-server](https://github.com/AMD-Ecosystem/triton-inference-server-server/tree/rocm10.0.0_r26.09) | `rocm10.0.0_r26.09` |
| Core | [AMD-Ecosystem/triton-inference-server-core](https://github.com/AMD-Ecosystem/triton-inference-server-core/tree/rocm10.0.0_r26.09) | `rocm10.0.0_r26.09` |
| Backend | [AMD-Ecosystem/triton-inference-server-backend](https://github.com/AMD-Ecosystem/triton-inference-server-backend/tree/rocm10.0.0_r26.09) | `rocm10.0.0_r26.09` |
| Third Party | [AMD-Ecosystem/triton-inference-server-third_party](https://github.com/AMD-Ecosystem/triton-inference-server-third_party/tree/rocm10.0.0_r26.09) | `rocm10.0.0_r26.09` |
| ONNX Runtime Backend | [AMD-Ecosystem/triton-inference-server-onnxruntime_backend](https://github.com/AMD-Ecosystem/triton-inference-server-onnxruntime_backend/tree/rocm10.0.0_r26.09) | `rocm10.0.0_r26.09` |
| Python Backend | [AMD-Ecosystem/triton-inference-server-python_backend](https://github.com/AMD-Ecosystem/triton-inference-server-python_backend/tree/rocm10.0.0_r26.09) | `rocm10.0.0_r26.09` |
| vLLM Backend | [AMD-Ecosystem/triton-inference-server-vllm_backend](https://github.com/AMD-Ecosystem/triton-inference-server-vllm_backend/tree/rocm10.0.0_r26.09) | `rocm10.0.0_r26.09` |
| Pytorch Backend | [AMD-Ecosystem/triton-inference-server-pytorch_backend](https://github.com/AMD-Ecosystem/triton-inference-server-pytorch_backend/tree/rocm10.0.0_r26.09) | `rocm10.0.0_r26.09` |
| Tensorflow Backend | [ROCm/triton-inference-server-tensorflow_backend](https://github.com/ROCm/triton-inference-server-tensorflow_backend/tree/rocm7.2.3_r24.03) | `rocm7.2.3_r24.03` |


## Build Triton Inference Server

### On Ubuntu 24.04 (ROCm 10.0.0, this branch)

**API impact: none.** Same `build.py` flags as NVIDIA container builds; no new kwargs or env vars.

This is the release recipe for **Ubuntu 24.04 + ROCm 10.0.0 + Triton + onnxruntime** (MIGraphX EP via AMD pip). `build.py` generates `Dockerfile.buildbase` / `Dockerfile` / `Dockerfile.cibase` at build time. Base image: `rocm/dev-ubuntu-24.04:10.0.0-full` (Composable Kernel is already in that tag).

#### Prerequisites

- Docker installed and running, with access to `/var/run/docker.sock` (nested ORT image)
- AMD GPU with ROCm 10.0.0 (or compatible) on the host
- Clone **this** branch from AMD-Ecosystem (`rocm10.0.0_r26.09`)

#### Optional thin base

Not required. Use only if you want a locally tagged image with extra apt deps and HIP on `LD_LIBRARY_PATH`:

```bash
git clone -b rocm10.0.0_r26.09 https://github.com/AMD-Ecosystem/triton-inference-server-server.git
cd triton-inference-server-server
bash scripts/build_ubuntu24.04_rocm_10_base.sh
# then: python3 build.py ... --image=base,localhost/ubuntu24.04_rocm10.0.0
```

#### Product image (onnxruntime)

```bash
cd triton-inference-server-server
python3 build.py \
  --enable-rocm \
  --linux-distro ubuntu \
  --no-container-interactive \
  --no-container-pull \
  --enable-logging \
  --enable-stats \
  --enable-metrics \
  --enable-cpu-metrics \
  --enable-tracing \
  --endpoint http \
  --endpoint grpc \
  --backend onnxruntime
```

Tags **`tritonserver:latest`**. Nested image `tritonserver_onnxruntime` installs **onnxruntime 1.29** and **MIGraphX 2.17** from AMD pip (`onnxruntime-ep-migraphx`).

Python, PyTorch, vLLM, and TensorFlow backends are **not** in this command. vLLM’s ROCm 10 wheel is cp314; this base is Python 3.12. TensorFlow stays on `rocm7.2.3_r24.03`.

**Build options:**
- `--enable-rocm`: HIP/ROCm (do not also pass CUDA `--enable-gpu`)
- `--linux-distro ubuntu`: Ubuntu 24.04 ROCm 10.0.0-full
- `--backend onnxruntime`: Triton ONNX Runtime backend + MIGraphX EP

#### Run

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --ipc=host \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -e LD_LIBRARY_PATH=/opt/rocm/lib \
  -v /path/to/model_repository:/models \
  tritonserver:latest \
  tritonserver --model-repository=/models
```



### On Debian 12

#### Prerequisites

- Docker installed and running
- AMD GPU with ROCm support
- ROCm 7.2.3 or compatible version installed on the host

The following instructions are for building on **Debian 12** with ROCm 7.2.3.

Step1: build base docker image with Debian12+ROCm7.2.3
```bash
git clone -b rocm7.2.3_r25.12 https://github.com/ROCm/triton-inference-server-server.git
cd triton-inference-server-server
bash scripts/build_debian12_rocm_723_base.sh
```
**OR** if vLLM engine is required (for vLLM backend or simply using vLLM engine in python backend)  
This creates the same base image for Debian12 distro, installing all depdencies and building vLLM from scratch.
```bash
cd triton-inference-server-server
bash scripts/build_debian12_rocm_72_vllm_base.sh
```


Step2: build tritonserver docker image
```bash
cd triton-inference-server-server
python3 build.py \
  --no-container-pull \
  --enable-logging \
  --enable-stats \
  --enable-tracing \
  --enable-rocm \
  --enable-metrics \
  --verbose \
  --endpoint=grpc \
  --endpoint=http \
  --backend=onnxruntime \
  --backend=python \
  --backend=vllm \
  --backend=tensorflow \
  --linux-distro=debian
```

**Build Options Explained:**
- `--enable-rocm`: Enable ROCm support
- `--linux-distro`: Build on Debian 12 OS
- `--endpoint=grpc --endpoint=http`: Enable both HTTP and gRPC inference protocols
- `--backend=onnxruntime`: Build with onnxruntime backend
- `--backend=python`: Build with python backend
- `--backend=vllm`: Build with vllm backend (vllm installed in base image)
- `--backend=tensorflow`: Build with tensorflow backend


*The above example builds tritonserver artifact with both onnxruntime, python, vllm and tensorflow backends.

## Run Triton Server

Start the Triton Server container with your model repository:

```bash
docker run \
  --name tritonserver_container \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  -it \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  --net=host \
  -e LD_LIBRARY_PATH=/opt/rocm/lib \
  -e ORT_MIGRAPHX_MODEL_CACHE_PATH=/migraphx_cache \
  -e ORT_MIGRAPHX_CACHE_PATH=/migraphx_cache \
  -v /path/to/your/model_repository/on/host:/models \
  -v /path/to/your/migraphx_cache_save_dir/on/host:/migraphx_cache \
  tritonserver:latest \
  tritonserver --model-repository=/models
```

**Important Parameters:**
- `--device=/dev/kfd --device=/dev/dri`: Grant access to AMD GPU devices
- `-p 8000:8000 -p 8001:8001 -p 8002:8002`: Expose HTTP (8000), gRPC (8001), and metrics (8002) ports
- `-v /path/to/your/model_repository/on/host:/models`: Mount your model repository where your python model file and config located. **Please make sure you only have model checkpoints and model config file under /path/to/your/model_repository/on/host**

#### Testing with Performance Analyzer

Use the Triton SDK container to run performance tests, by default test requests are sent to localhost so please run the test client container on the same machine

```bash
run performance analyzer
perf_analyzer -m <model_name>  --input-data=<your input data file>
```

---
