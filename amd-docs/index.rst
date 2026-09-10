.. meta::
  :description: Triton Inference Server documentation
  :keywords: Triton Inference Server, ROCm, documentation, deep learning, framework, GPU

.. _triton-inference-server-documentation-index:

********************************************************************
Triton Inference Server on ROCm documentation
********************************************************************

Use Triton Inference Server on ROCm to serve large language models, computer vision models,
recommender systems, and custom pipelines, all from a single platform.

Triton Inference Server is a high-performance model server for machine learning inference.
It supports a wide range of model types and deep learning frameworks, including
Pytorch, Tensorflow, ONNX runtime, Python, vLLM and more. Triton Inference
Server handles concurrent model execution, dynamic batching, model ensembles, and streaming
inference, maximizing throughput and GPU/CPU utilization for production deployments at scale.

Triton Inference Server on ROCm is optimized by the AMD ROCm software stack for high performance on
AMD Instinct GPUs. It integrates ROCm-aware runtime libraries and optimized kernel backends,
delivering efficient inference across all supported model types and frameworks on AMD hardware.

.. note::

  The ROCm port of Triton Inference Server is under active development, and some features are not yet available. 
  For the most up-to-date feature support, refer to the ``README`` in the 
  `https://github.com/ROCm/triton-inference-server-server <https://github.com/ROCm/triton-inference-server-server>`__ repository.

Triton Inference Server is part of the `ROCm-LLMExt toolkit
<https://rocm.docs.amd.com/projects/rocm-llmext/en/docs-26.03/>`__.

The Triton Inference Server public repository is located at `https://github.com/ROCm/triton-inference-server-server <https://github.com/ROCm/triton-inference-server-server>`__.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install Triton Inference Server <install/triton-inference-server-install>`

  .. grid-item-card:: Reference

      * `User guide and API reference (upstream) <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html>`__

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the :doc:`Licensing <about/license>` page.
