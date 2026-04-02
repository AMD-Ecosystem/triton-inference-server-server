.. meta::
  :description: installing Triton Inference Server for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, Triton Inference Server

.. _triton-inference-server-on-rocm-installation:

********************************************************************
Triton Inference Server on ROCm installation
********************************************************************

System requirements
====================================================================

To use Triton Inference Server `25.12 <https://github.com/ROCm/triton-inference-server-server/tree/release/r25.12%2Bamd.1>`__, you need the following prerequisites:

- **ROCm version:** `7.2.0 <https://rocm.docs.amd.com/en/docs-7.2.0/>`__
- **Operating system:** Ubuntu 24.04
- **GPU platform:** AMD Instinct™ MI300X, MI355X
- **Python:** `3.12 <https://www.python.org/downloads/release/python-3129/>`__

Install Triton Inference Server
================================================================================

To install Triton Inference Server on ROCm, you have the following options:

* :ref:`using-docker-with-triton-inference-server-pre-installed` **(recommended)**
* :ref:`build-triton-inference-server-rocm-docker-image`

.. _using-docker-with-triton-inference-server-pre-installed:

Use a prebuilt Docker image with Triton Inference Server pre-installed
--------------------------------------------------------------------------------------

Docker is the recommended method to set up a Triton Inference Server environment for model serving,
as it avoids dependency conflicts.  The tested, prebuilt image includes Triton Inference Server, Python, 
ROCm, and all other requirements.

1. Pull the Docker image; this docker image is built on Ubuntu 24.04 and ROCm 7.2.0 with ONNX runtime and Python backends enabled.

   .. code-block:: bash

      docker pull rocm/tritoninferenceserver:tritoninferenceserver-25.12.amd1_rocm7.2_ubuntu24.04_py3.12

2. Start a Docker container using the image.

   .. code-block:: bash

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
      -e ORT_MIGRAPHX_MODEL_CACHE_PATH=/migraphx_cache \
      -e ORT_MIGRAPHX_CACHE_PATH=/migraphx_cache \
      -v /path/to/your/model_repository/on/host:/models \
      -v /path/to/your/migraphx_cache_save_dir/on/host:/migraphx_cache \
      tritoninferenceserver:tritoninferenceserver-25.12.amd1_rocm7.2_ubuntu24.04_py3.12 \
      tritonserver --model-repository=/models --exit-on-error=false

3. The prebuilt image contains the Triton Server executable, required shared libraries, backends, and repository agents in the following locations:

   - Triton Inference Server executable: ``/opt/tritonserver/bin``
   - Shared libraries: ``/opt/tritonserver/lib``
   - Backends: ``/opt/tritonserver/backends``
   - Repository agents: ``/opt/tritonserver/repoagents``

.. _build-triton-inference-server-rocm-docker-image:

Build from source
--------------------------------------------------------------------------------------

Triton Inference Server on ROCm can be run directly by setting up a Docker container from scratch.

1. Clone the `https://github.com/ROCm/triton-inference-server-server <https://github.com/ROCm/triton-inference-server-server>`__ repository and enter the directory.

   .. code-block:: bash
      
      git clone -b rocm7.2_r25.12 https://github.com/ROCm/triton-inference-server-server.git
      cd triton-inference-server-server
      bash scripts/build_ubuntu24.04_rocm_72_base.sh

2. Build the Docker image.
   
   .. code-block:: bash
      
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
      --linux-distro=ubuntu

3. Ensure your build options are set as follows:

   - ``--enable-rocm``: Enable ROCm support.
   - ``--endpoint``: Build with HTTP and gRPC endpoints.
   - ``--backend=onnxruntime`` / ``--backend=python``: Build backends into the server.
   - ``--linux-distro``: Build on Ubuntu 24.04.

4. The above settings build a Triton server with both ONNX Runtime and Python backends enabled.
After the build completes, you can run the Docker container using the same command shown in the
:ref:`using-docker-with-triton-inference-server-pre-installed` section.


Test the Triton Inference Server installation
======================================================================================

After launching Triton using the ``docker run`` command, you should see the model repository load
successfully. Your models will also show as "READY" in the server logs.
