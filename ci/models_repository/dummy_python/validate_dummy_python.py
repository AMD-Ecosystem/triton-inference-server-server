#!/usr/bin/env python3
"""
Validate the dummy_python model: run reference using numpy logic as model.py
and compare with Triton Inference Server directly. Uses regular HTTP requests.
"""
import sys
import numpy as np
import requests

INPUT_A_NAME = "INPUT_A"
INPUT_B_NAME = "INPUT_B"
OUTPUT_NAMES = ["SUM", "DIFF", "DOT_PRODUCT", "L2_NORM_A", "CONCAT"]
INPUT_DIM = 4  # each input is [batch, 4]


def run_python_reference(batch_size: int, seed: int = 123):
    """
    Run the same logic as dummy_python model.py (CPU, numpy only).
    Returns (input_a, input_b, dict of output_name -> array).
    """
    np.random.seed(seed)
    input_a = np.random.randn(batch_size, INPUT_DIM).astype(np.float32)
    input_b = np.random.randn(batch_size, INPUT_DIM).astype(np.float32)

    sum_out = (input_a + input_b).astype(np.float32)
    diff_out = (input_a - input_b).astype(np.float32)
    dot_out = np.sum(input_a * input_b, axis=1, keepdims=True).astype(np.float32)
    l2_out = np.sqrt(np.sum(input_a * input_a, axis=1, keepdims=True)).astype(np.float32)
    concat_out = np.concatenate([input_a, input_b], axis=1).astype(np.float32)

    outputs = {
        "SUM": sum_out,
        "DIFF": diff_out,
        "DOT_PRODUCT": dot_out,
        "L2_NORM_A": l2_out,
        "CONCAT": concat_out,
    }
    return input_a, input_b, outputs


def _triton_base_url(url: str) -> str:
    """Ensure URL has http scheme for Triton server."""
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        return "http://" + url
    return url


def _triton_is_live(base_url: str) -> bool:
    """Return True if Triton server is live."""
    try:
        r = requests.get(f"{base_url}/v2/health/live", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _triton_model_index(base_url: str, ready_only: bool = False) -> list:
    """Return repository index: list of {"name", "state", ...}. ready_only=True returns only READY models."""
    try:
        r = requests.post(
            f"{base_url}/v2/repository/index",
            json={"ready": ready_only},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _require_models_ready(base_url: str, required_names: list) -> None:
    """Ensure all required models are READY; exit with clear error if any is missing or not READY."""
    index = _triton_model_index(base_url, ready_only=False)
    by_name = {m["name"]: m for m in index if isinstance(m, dict) and "name" in m}
    for name in required_names:
        if name not in by_name:
            print(f"ERROR: Model '{name}' not in repository index", file=sys.stderr)
            sys.exit(1)
        state = by_name[name].get("state", "")
        if state != "READY":
            reason = by_name[name].get("reason", "")
            msg = f"ERROR: Model '{name}' is not READY (status: {state!r})"
            if reason:
                msg += f"; reason: {reason}"
            print(msg, file=sys.stderr)
            sys.exit(1)


def run_triton_inference(url: str, model_name: str, input_a: np.ndarray, input_b: np.ndarray):
    """Run inference via Triton HTTP API; return dict of output_name -> array."""
    base = _triton_base_url(url)
    infer_url = f"{base}/v2/models/{model_name}/infer"
    payload = {
        "inputs": [
            {
                "name": INPUT_A_NAME,
                "shape": list(input_a.shape),
                "datatype": "FP32",
                "data": input_a.flatten().tolist(),
            },
            {
                "name": INPUT_B_NAME,
                "shape": list(input_b.shape),
                "datatype": "FP32",
                "data": input_b.flatten().tolist(),
            },
        ],
        "outputs": [{"name": name} for name in OUTPUT_NAMES],
    }
    r = requests.post(infer_url, json=payload, timeout=30)
    r.raise_for_status()
    out = r.json()
    result = {}
    for o in out.get("outputs", []):
        name = o.get("name")
        if name in OUTPUT_NAMES:
            data = o.get("data")
            shape = o.get("shape", [])
            result[name] = np.array(data, dtype=np.float32).reshape(shape)
    if len(result) != len(OUTPUT_NAMES):
        missing = set(OUTPUT_NAMES) - set(result)
        raise KeyError(f"Output(s) not in response: {missing}")
    return result


def validate_against_triton(
    url: str = "localhost:8000",
    model_name: str = "dummy_python",
    batch_sizes=(1, 4, 8),
    seed: int = 123,
):
    """Compare Python reference output with Triton for the same inputs (in memory)."""
    rtol, atol = 1e-5, 1e-5
    base = _triton_base_url(url)

    try:
        if not _triton_is_live(base):
            print(f"ERROR: Triton server not live at {url}", file=sys.stderr)
            sys.exit(1)
        _require_models_ready(base, [model_name])
    except Exception as e:
        print(f"ERROR: Cannot connect to Triton at {url}: {e}", file=sys.stderr)
        sys.exit(1)

    all_passed = True
    for batch_size in batch_sizes:
        input_a, input_b, ref_outputs = run_python_reference(batch_size, seed=seed)
        triton_outputs = run_triton_inference(url, model_name, input_a, input_b)
        for name in OUTPUT_NAMES:
            ref_arr = ref_outputs[name]
            triton_arr = triton_outputs[name]
            match = np.allclose(triton_arr, ref_arr, rtol=rtol, atol=atol)
            if not match:
                diff = np.abs(triton_arr - ref_arr)
                print(f"  batch_size={batch_size} {name}: MISMATCH (max_diff={diff.max():.6e})")
                all_passed = False
            else:
                print(f"  batch_size={batch_size} {name}: OK")
    if all_passed:
        print("Triton validation: all passed.")
    else:
        print("Triton validation: some comparisons failed.", file=sys.stderr)
        sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate dummy_python model against Triton (reference vs server, direct comparison)",
    )
    parser.add_argument(
        "--triton-url",
        default="localhost:8000",
        help="Triton server URL (default: localhost:8000)",
    )
    parser.add_argument(
        "--model-name",
        default="dummy_python",
        help="Triton model name (default: dummy_python)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Batch sizes to test (default: 1 4 8)",
    )
    args = parser.parse_args()

    validate_against_triton(
        url=args.triton_url,
        model_name=args.model_name,
        batch_sizes=tuple(args.batch_sizes),
    )


if __name__ == "__main__":
    main()
