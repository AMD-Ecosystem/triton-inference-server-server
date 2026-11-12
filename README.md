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

## This branch (`rocm10.1_r25.12`)

This checkout is **not** the stock ROCm 7.2.3 README flow. It is the
AMD-Ecosystem ROCm 10.1 branch with:

- Debian 12 + **ROCm 10.1.0** (ROCm Core SDK) as the default container OS
- MIGraphX **plugin EP output-name remap** as the default ONNX Runtime EP
- **PyTorch** and **TensorFlow** backends enabled for that 10.1 path

Clone this tree, not `ROCm/triton-inference-server-server` @ `rocm7.2.3_r25.12`:

```bash
git clone -b rocm10.1_r25.12 \
  https://github.com/AMD-Ecosystem/triton-inference-server-server.git
cd triton-inference-server-server
```

The Ubuntu 24.04 / ROCm 7.2.3 instructions later in this file are leftover from
the upstream README. They still work from the scripts in-tree, but they are
**not** what `build.py` uses by default on Debian.

## ROCm Support

This repository contains ROCm-enabled builds of Triton Inference Server for AMD GPUs. The following backends are available or in development:

- **ONNX Runtime Backend** - Done
- **Python Backend** - Done
- **vLLM Backend** - Done (ROCm 7.2.3 path; not the default 10.1 Debian image)
- **PyTorch Backend**  - Done
- **Tensorflow Backend** - Done

### ROCm 10.1 pins used by this branch

`build.py --enable-rocm --linux-distro=debian` currently uses:

| Component | Repository | Branch / tag / image |
|-----------|------------|----------------------|
| Server (this repo) | [AMD-Ecosystem/triton-inference-server-server](https://github.com/AMD-Ecosystem/triton-inference-server-server) | `rocm10.1_r25.12` |
| Core | [ROCm/triton-inference-server-core](https://github.com/ROCm/triton-inference-server-core) | `rocm7.2_r25.12` |
| Backend | [ROCm/triton-inference-server-backend](https://github.com/ROCm/triton-inference-server-backend) | `rocm7.2_r25.12` |
| Third Party | [ROCm/triton-inference-server-third_party](https://github.com/ROCm/triton-inference-server-third_party) | `rocm7.2_r25.12` |
| ONNX Runtime Backend | [TedThemistokleous/onnxruntime_backend](https://github.com/TedThemistokleous/onnxruntime_backend) | `pre_verify_10.1` |
| MIGraphX plugin EP | [ffloresy/onnxruntime-ep-amdgpu](https://github.com/ffloresy/onnxruntime-ep-amdgpu) | `fix-output-name-remap` |
| MIGraphX (source) | [ROCm/AMDMIGraphX](https://github.com/ROCm/AMDMIGraphX) | `remove_output_concats2` |
| Python Backend | [ROCm/triton-inference-server-python_backend](https://github.com/ROCm/triton-inference-server-python_backend) | `rocm7.2_r25.12` |
| Pytorch Backend | [ROCm/triton-inference-server-pytorch_backend](https://github.com/ROCm/triton-inference-server-pytorch_backend) | `rocm7.2_r25.12` |
| Tensorflow Backend | [ROCm/triton-inference-server-tensorflow_backend](https://github.com/ROCm/triton-inference-server-tensorflow_backend) | `rocm7.14-prev_r24.03` |
| TensorFlow ROCm image | local Docker image | `localhost/debian12_rocm10.1.0_tensorflow` |

Override the plugin EP with `--migraphx-ep-repo` / `--migraphx-ep-branch` if you
need the upstream EP instead of the name-remap fork.

## Build Triton Inference Server (ROCm 10.1, Debian 12) — this branch

#### Prerequisites

- Docker installed and running
- AMD GPU with ROCm support (images currently install **gfx942** packages: MI300X / MI325X / MI300A)
- Host ROCm 10.1-compatible driver

ROCm 10.1 uses the **ROCm Core SDK** repo (`stable.repo.amd.com/rocm/core`), not
the legacy ROCm 7.x `repo.amd.com` layout.

### Step 1 — Debian 12 + ROCm 10.1 base image

```bash
bash scripts/build_debian12_rocm_101_base.sh
```

This builds `localhost/debian12_rocm10.1.0`, which `build.py --linux-distro=debian` uses as the base.

### Step 2 — TensorFlow ROCm 10.1 image (required for `--backend=tensorflow`)

```bash
./scripts/build_debian12_rocm_101_tensorflow.sh
```

This builds `localhost/debian12_rocm10.1.0_tensorflow`. `build.py` passes that
name to the TensorFlow backend as `TRITON_TENSORFLOW_DOCKER_IMAGE`.

What the image does:

- Clones [ROCm/tensorflow-upstream](https://github.com/ROCm/tensorflow-upstream) @ `develop-upstream`
- Applies `patches/tensorflow_tritonapi_build_develop-upstream_adjusted.patch`
- Builds the `tensorflow_rocm` wheel with Bazel (hermetic Python 3.12) and checks that TF can see a GPU

The patch injects Triton’s TensorFlow C API (`tensorflow_backend_tf`) into that
TF tree. GPU-vs-CPU tensor detection uses TensorFlow’s `GetMemoryType()` API
instead of HIP (`hipPointerGetAttributes`), because HIP headers are not visible
in this hermetic Bazel sandbox.

Skip the long TF compile during `docker build` if you only need the image shell:

```bash
RUN_TF_REPRO_DURING_BUILD=0 ./scripts/build_debian12_rocm_101_tensorflow.sh
```

**GPU target note:** the TensorFlow Dockerfile currently sets
`TF_ROCM_AMDGPU_TARGETS=gfx942` for a faster validation build. Restore the full
target list in `Dockerfile.debian12_rocm10.1.0_tensorflow` if you need other
architectures.

### Step 3 — build `tritonserver`

```bash
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
  --backend=pytorch \
  --backend=tensorflow \
  --linux-distro=debian
```

**Build options explained:**
- `--enable-rocm`: Enable ROCm support
- `--linux-distro=debian`: Use `localhost/debian12_rocm10.1.0`
- `--backend=onnxruntime`: ONNX Runtime + MIGraphX plugin EP (`fix-output-name-remap` by default)
- `--backend=python`: Python backend
- `--backend=pytorch`: PyTorch backend
- `--backend=tensorflow`: TensorFlow backend (needs Step 2)

Do not add `--backend=vllm` on this 10.1 path unless you have also built a
matching vLLM base image. The in-tree vLLM base scripts are still ROCm 7.2.

## Run Triton Server

Start the Triton Server container with your model repository. The `ORT_MIGRAPHX_*`
and `MIGRAPHX_*` environment variables apply to **ONNX Runtime + MIGraphX**
models only (not PyTorch or TensorFlow).

```bash
docker run \
  --name tritonserver_container \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --net=host \
  -it \
  -e ROCR_VISIBLE_DEVICES=0 \
  -e MIGRAPHX_MLIR_USE_SPECIFIC_OPS="attention,convolution,dot,fused" \
  -e ORT_MIGRAPHX_MODEL_CACHE_PATH=/models/cache/ \
  -e ORT_MIGRAPHX_HIP_GRAPH_ENABLE=1 \
  -e ORT_MIGRAPHX_COALESCE_IO=1 \
  -e ORT_MIGRAPHX_COMPILE_BATCHES="1,2,4,8,16,32,64,128" \
  -v /path/to/your/model_repository/on/host:/models \
  -v /path/to/your/migraphx_cache_save_dir/on/host:/models/cache \
  tritonserver \
  tritonserver --model-repository=/models --exit-on-error=false
```

**Important parameters:**
- `--device=/dev/kfd --device=/dev/dri`: Grant access to AMD GPU devices
- `--group-add video --ipc=host --net=host`: GPU device group, shared memory, and host networking
- `-v /path/to/your/model_repository/on/host:/models`: Mount your model repository. **Please make sure you only have model checkpoints and model config file under that host path.**
- `-v /path/to/your/migraphx_cache_save_dir/on/host:/models/cache`: Mount the compiled-model (`.mxr`) cache. Keep this directory **outside** the model repository tree if you use `--model-control-mode=explicit`, so Triton does not treat the cache as a model.

`ORT_MIGRAPHX_COALESCE_IO=1` requires `ORT_MIGRAPHX_HIP_GRAPH_ENABLE=1`; otherwise
it is ignored. If you change `MIGRAPHX_MLIR_USE_SPECIFIC_OPS`, delete the `*.mxr`
files in the cache directory or the old compiled graphs are reused.

### Environment variable references

The last four `ORT_MIGRAPHX_*` names are defined by the **MIGraphX plugin EP**
(`onnxruntime-ep-amdgpu`), which this branch builds by default. The public
[onnxruntime.ai MIGraphX page](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)
documents `ORT_MIGRAPHX_MODEL_CACHE_PATH` but not `HIP_GRAPH_ENABLE`,
`COALESCE_IO`, or `COMPILE_BATCHES`.

| Variable | What it does | Reference |
|----------|--------------|-----------|
| `ROCR_VISIBLE_DEVICES` | ROCm runtime GPU isolation: only the listed GPU index/UUID is visible. On Linux this is preferred over `HIP_VISIBLE_DEVICES`. | [ROCm environment variables](https://rocm.docs.amd.com/en/latest/reference/environment-variables/index.html), [GPU isolation techniques](https://rocm.docs.amd.com/en/latest/reference/system-optimization/gpu-isolation.html), [ROCR-Runtime env vars](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/develop/api-reference/environment_variables.html) |
| `MIGRAPHX_MLIR_USE_SPECIFIC_OPS` | Forces listed ops through rocMLIR (`attention,convolution,dot,fused`). No spaces in the list. Prefix with `~` to disable an op. | [MIGraphX environment variables](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/reference/MIGraphX-dev-env-vars.html), [Tune model performance](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/how-to/model-performance-tuning.html) |
| `ORT_MIGRAPHX_MODEL_CACHE_PATH` | Directory for compiled MIGraphX `.mxr` files so later starts skip recompilation. | [Plugin EP env-var table](https://github.com/onnxruntime/onnxruntime-ep-amdgpu/blob/main/docs/MIGraphX-ExecutionProvider.md#environment-variables), [onnxruntime.ai MIGraphX EP](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html), [AMD blog — Triton + ONNX](https://rocm.blogs.amd.com/software-tools-optimization/triton-server-onnx/README.html) |
| `ORT_MIGRAPHX_HIP_GRAPH_ENABLE` | Enable HIP graph capture/replay for the compiled MIGraphX model (`1`/`0`). Plugin EP, added in ROCm 7.2. | [Plugin EP env-var table](https://github.com/onnxruntime/onnxruntime-ep-amdgpu/blob/main/docs/MIGraphX-ExecutionProvider.md#environment-variables), [source: `mgx_ep.h`](https://github.com/onnxruntime/onnxruntime-ep-amdgpu/blob/main/src/migraphx/mgx_ep.h) |
| `ORT_MIGRAPHX_COALESCE_IO` | Pack all host inputs into one H2D copy on the pinned path. Requires `ORT_MIGRAPHX_HIP_GRAPH_ENABLE=1`. Plugin EP, added in ROCm 7.14. | [Plugin EP env-var table](https://github.com/onnxruntime/onnxruntime-ep-amdgpu/blob/main/docs/MIGraphX-ExecutionProvider.md#environment-variables), [source: `mgx_ep.h`](https://github.com/onnxruntime/onnxruntime-ep-amdgpu/blob/main/src/migraphx/mgx_ep.h) |
| `ORT_MIGRAPHX_COMPILE_BATCHES` | Comma-separated batch sizes to pre-compile (e.g. `"1,2,4,8,16,32,64,128"`). Fewer sizes means a faster first compile. Plugin EP, added in ROCm 7.2. | [Plugin EP env-var table](https://github.com/onnxruntime/onnxruntime-ep-amdgpu/blob/main/docs/MIGraphX-ExecutionProvider.md#environment-variables), [source: `mgx_ep.h`](https://github.com/onnxruntime/onnxruntime-ep-amdgpu/blob/main/src/migraphx/mgx_ep.h), [AMD blog `docker run` example](https://rocm.blogs.amd.com/software-tools-optimization/triton-server-onnx/README.html) |

#### Testing with Performance Analyzer

Use the Triton SDK container to run performance tests. By default, test requests are sent to localhost, so run the test client container on the same machine.

```bash
perf_analyzer -m <model_name>  --input-data=<your input data file>
```

---

## Upstream leftover: Ubuntu 24.04 / Debian 12 with ROCm 7.2.3

These scripts are still in the tree, but they are **not** this branch’s default.
They clone/build the older ROCm 7.2.3 layout (`localhost/ubuntu24.04_rocm7.2.3`
or `localhost/debian12_rocm7.2.3`).

### On Ubuntu 24.04 (ROCm 7.2.3)

```bash
bash scripts/build_ubuntu24.04_rocm_723_base.sh
python3 build.py \
  --no-container-pull \
  --enable-logging --enable-stats --enable-tracing \
  --enable-rocm --enable-metrics --verbose \
  --endpoint=grpc --endpoint=http \
  --backend=onnxruntime --backend=python \
  --backend=vllm --backend=pytorch --backend=tensorflow \
  --linux-distro=ubuntu
```

### On Debian 12 (ROCm 7.2.3)

```bash
bash scripts/build_debian12_rocm_723_base.sh
# optional vLLM base:
# bash scripts/build_debian12_rocm_72_vllm_base.sh
python3 build.py \
  --no-container-pull \
  --enable-logging --enable-stats --enable-tracing \
  --enable-rocm --enable-metrics --verbose \
  --endpoint=grpc --endpoint=http \
  --backend=onnxruntime --backend=python \
  --backend=vllm --backend=tensorflow \
  --linux-distro=debian
```
