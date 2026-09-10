.. meta::
  :description: Triton Inference Server examples
  :keywords: Triton Inference Server, programming, ROCm, example, sample, tutorial, MI300X, MI355X, ONNX Runtime, MIGraphX, perf_analyzer

.. _run-a-triton-inference-server-example:

********************************************************************
Run a Triton Inference Server example
********************************************************************

This example shows how to deploy and run a simple ONNX model using Triton Inference Server ONNX runtime backend MIGraphX execution provider on AMD Instinct MI300X or MI355X GPUs.

1. Prepare a Triton model repository with a simple ONNX model.

   Create a dummy two-layer MLP ONNX model and a Triton model repository layout that uses the ONNX Runtime backend with the MIGraphX execution provider.

   .. code-block:: bash

      # Create the model repository directory
      mkdir -p $PWD/model_repository/dummy_migraphx_onnx/0

      # Write the Triton model configuration
      cat > $PWD/model_repository/dummy_migraphx_onnx/config.pbtxt << 'EOF'
      name: "dummy_migraphx_onnx"
      backend: "onnxruntime"
      max_batch_size: 64

      dynamic_batching {
          max_queue_delay_microseconds: 300
          preserve_ordering: true
      }

      optimization {
          cuda {
              graphs: true
          }
          input_pinned_memory {
              enable: true
          }
          execution_accelerators {
              gpu_execution_accelerator {
                  name: "migraphx"
                  parameters: {
                      key: "migraphx_max_dynamic_batch"
                      value: "64"
                  }
              }
          }
      }

      input: [
          { name: "input", data_type: TYPE_FP32, dims: [8] }
      ]
      output: [
          { name: "output", data_type: TYPE_FP32, dims: [4] }
      ]

      instance_group [
          { kind: KIND_GPU, count: 2, gpus: [0] }
      ]
      EOF

   Save the following Python script to generate a dummy two-layer MLP ONNX model (input dimension 8, hidden dimension 16, output dimension 4) with dynamic batching support.

   .. code-block:: python

      # save as create_dummy_onnx.py under your current working directory (run with: python create_dummy_onnx.py)
      import os
      import numpy as np
      import onnx
      from onnx import helper, TensorProto

      INPUT_SIZE = 8
      HIDDEN_SIZE = 16
      OUTPUT_SIZE = 4

      MODEL_DIR = os.path.join(os.getcwd(), "model_repository", "dummy_migraphx_onnx", "0")
      MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
      os.makedirs(MODEL_DIR, exist_ok=True)

      x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", INPUT_SIZE])
      y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", OUTPUT_SIZE])

      np.random.seed(42)
      W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE).astype(np.float32) * 0.1
      b1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
      W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE).astype(np.float32) * 0.1
      b2 = np.zeros(OUTPUT_SIZE, dtype=np.float32)

      w1 = helper.make_tensor("W1", TensorProto.FLOAT, [INPUT_SIZE, HIDDEN_SIZE], W1.flatten().tolist())
      b1_t = helper.make_tensor("b1", TensorProto.FLOAT, [HIDDEN_SIZE], b1.tolist())
      w2 = helper.make_tensor("W2", TensorProto.FLOAT, [HIDDEN_SIZE, OUTPUT_SIZE], W2.flatten().tolist())
      b2_t = helper.make_tensor("b2", TensorProto.FLOAT, [OUTPUT_SIZE], b2.tolist())

      nodes = [
          helper.make_node("MatMul", ["input", "W1"], ["h1"], name="matmul1"),
          helper.make_node("Add", ["h1", "b1"], ["h1b"], name="add1"),
          helper.make_node("Relu", ["h1b"], ["h1r"], name="relu1"),
          helper.make_node("MatMul", ["h1r", "W2"], ["h2"], name="matmul2"),
          helper.make_node("Add", ["h2", "b2"], ["output"], name="add2"),
      ]

      graph = helper.make_graph(
          nodes, "dummy_mlp", [x], [y],
          initializer=[w1, b1_t, w2, b2_t],
      )
      model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
      model.ir_version = 9
      onnx.checker.check_model(model)
      onnx.save(model, MODEL_PATH)
      print(f"Saved: {MODEL_PATH}")
      print(f"Input shape: [batch, {INPUT_SIZE}], Output shape: [batch, {OUTPUT_SIZE}]")

   .. code-block:: bash

      # Generate the ONNX model file in the model repository
      python create_dummy_onnx.py
      ls -R $PWD/model_repository/

2. Start Triton Inference Server (ROCm-enabled) with your model repository.

   Use a ROCm-enabled Triton container and pass through the AMD GPU devices. The example below assumes a ROCm 7.2 + Triton 25.12 image tag.

   .. code-block:: bash

      # Pull the ROCm-enabled Triton Server image
      docker pull rocm/tritoninferenceserver:tritoninferenceserver-25.12.amd1_rocm7.2_ubuntu24.04_py3.12

      # Create a host directory for MIGraphX compilation cache
      mkdir -p $PWD/migraphx_cache

      # Launch Triton Server, mapping your model repository and MIGraphX cache
      docker run --rm -it --net=host \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video \
        --ipc=host \
        -p 8000:8000 \
        -p 8001:8001 \
        -p 8002:8002 \
        -e ORT_MIGRAPHX_MODEL_CACHE_PATH=/migraphx_cache \
        -e ORT_MIGRAPHX_CACHE_PATH=/migraphx_cache \
        -v $PWD/model_repository:/models \
        -v $PWD/migraphx_cache:/migraphx_cache \
        rocm/tritoninferenceserver:tritoninferenceserver-25.12.amd1_rocm7.2_ubuntu24.04_py3.12 \
        tritonserver --model-repository=/models --exit-on-error=false

   Keep this terminal running. You should see Triton load the ``dummy_migraphx_onnx`` model and report that the model is ready.

3. Use the python script or ``perf_analyzer`` tool from the `https://github.com/ROCm/triton-inference-server-server <https://github.com/ROCm/triton-inference-server-server>`__ repository to send client requests for end-to-end benchmark tests.
