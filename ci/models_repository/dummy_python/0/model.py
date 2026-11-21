"""
Dummy Python backend model for Triton: multi-input, multi-output model with
batched execution, shape validation, and stateful request counting.
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Triton Python model class. Class name must be TritonPythonModel."""

    def initialize(self, args):
        """Called once when the model is loaded. Parses config and output dtypes."""
        self.model_config = json.loads(args["model_config"])
        self.model_name = args.get("model_name", "dummy_python")
        self._request_count = 0

        # Resolve output dtypes from config
        self._output_dtypes = {}
        for out_cfg in self.model_config.get("output", []):
            name = out_cfg["name"]
            self._output_dtypes[name] = pb_utils.triton_string_to_numpy(
                out_cfg["data_type"]
            )

        self._max_batch_size = self.model_config.get("max_batch_size", 0)
        self._support_batching = self._max_batch_size > 0

        if hasattr(pb_utils, "Logger") and pb_utils.Logger:
            pb_utils.Logger.log_info(
                f"[{self.model_name}] Initialized; max_batch_size={self._max_batch_size}"
            )

    def execute(self, requests):
        """
        Process batched inference:
          SUM         = INPUT_A + INPUT_B
          DIFF        = INPUT_A - INPUT_B
          DOT_PRODUCT = dot(INPUT_A, INPUT_B) as shape [1]
          L2_NORM_A   = L2 norm of INPUT_A as shape [1]
          CONCAT      = concat(INPUT_A, INPUT_B) -> length 8
        """
        responses = []
        for request in requests:
            self._request_count += 1
            err = self._validate_request(request)
            if err is not None:
                responses.append(
                    pb_utils.InferenceResponse(output_tensors=[], error=err)
                )
                continue

            input_a = pb_utils.get_input_tensor_by_name(request, "INPUT_A")
            input_b = pb_utils.get_input_tensor_by_name(request, "INPUT_B")
            a = input_a.as_numpy()
            b = input_b.as_numpy()

            # Ensure 2D [batch, 4] for batching
            if a.ndim == 1:
                a = np.expand_dims(a, axis=0)
            if b.ndim == 1:
                b = np.expand_dims(b, axis=0)
            batch_size = a.shape[0]

            sum_out = (a + b).astype(self._output_dtypes["SUM"])
            diff_out = (a - b).astype(self._output_dtypes["DIFF"])
            dot_out = np.sum(a * b, axis=1, keepdims=True).astype(
                self._output_dtypes["DOT_PRODUCT"]
            )
            l2_out = np.sqrt(np.sum(a * a, axis=1, keepdims=True)).astype(
                self._output_dtypes["L2_NORM_A"]
            )
            concat_out = np.concatenate([a, b], axis=1).astype(
                self._output_dtypes["CONCAT"]
            )

            response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("SUM", sum_out),
                    pb_utils.Tensor("DIFF", diff_out),
                    pb_utils.Tensor("DOT_PRODUCT", dot_out),
                    pb_utils.Tensor("L2_NORM_A", l2_out),
                    pb_utils.Tensor("CONCAT", concat_out),
                ]
            )
            responses.append(response)

        return responses

    def _validate_request(self, request):
        """Check that required inputs exist and have compatible shapes. Returns TritonError or None."""
        input_a = pb_utils.get_input_tensor_by_name(request, "INPUT_A")
        input_b = pb_utils.get_input_tensor_by_name(request, "INPUT_B")
        if input_a is None:
            return pb_utils.TritonError("INPUT_A tensor is missing")
        if input_b is None:
            return pb_utils.TritonError("INPUT_B tensor is missing")
        a = input_a.as_numpy()
        b = input_b.as_numpy()
        if a.shape != b.shape:
            return pb_utils.TritonError(
                f"INPUT_A shape {a.shape} does not match INPUT_B shape {b.shape}"
            )
        if a.shape[-1] != 4:
            return pb_utils.TritonError(
                f"Last dimension must be 4, got {a.shape}"
            )
        return None

    def finalize(self):
        """Cleanup: log total requests processed."""
        if hasattr(pb_utils, "Logger") and pb_utils.Logger:
            pb_utils.Logger.log_info(
                f"[{self.model_name}] Finalized; total requests={self._request_count}"
            )
