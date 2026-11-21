#!/usr/bin/env python3
"""
Validate the dummy ONNX model, uses regular HTTP requests.
"""
import os
import sys
import numpy as np
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "0")
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")

INPUT_SIZE = 8
OUTPUT_SIZE = 4
INPUT_NAME = "input"
OUTPUT_NAME = "output"

# generate reference input and output using onnx and CPU EP
def run_onnx_reference(model_path: str, batch_size: int, seed: int = 123):
    """Run model with ONNX Runtime; return (input_array, output_array)."""
    import onnxruntime as ort

    np.random.seed(seed)
    input_data = np.random.randn(batch_size, INPUT_SIZE).astype(np.float32)

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    out = session.run(None, {INPUT_NAME: input_data})[0]
    return input_data, out


def _triton_base_url(url: str) -> str:
    """Ensure URL has http scheme for Triton server."""
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        return "http://" + url
    return url


def run_triton_inference(url: str, model_name: str, input_data: np.ndarray):
    """Run inference via Triton HTTP API; return output array."""
    base = _triton_base_url(url)
    infer_url = f"{base}/v2/models/{model_name}/infer"
    payload = {
        "inputs": [
            {
                "name": INPUT_NAME,
                "shape": list(input_data.shape),
                "datatype": "FP32",
                "data": input_data.flatten().tolist(),
            }
        ],
        "outputs": [{"name": OUTPUT_NAME}],
    }
    r = requests.post(infer_url, json=payload, timeout=30)
    r.raise_for_status()
    out = r.json()
    for o in out.get("outputs", []):
        if o.get("name") == OUTPUT_NAME:
            data = o.get("data")
            shape = o.get("shape", [])
            return np.array(data, dtype=np.float32).reshape(shape)
    raise KeyError(f"Output '{OUTPUT_NAME}' not in response")


def validate_reference(batch_sizes=(1, 4, 8), seed=123):
    """Run ONNX reference for given batch sizes; save and print results."""
    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}", file=sys.stderr)
        print("Run create_dummy_onnx.py first.", file=sys.stderr)
        sys.exit(1)

    ref_dir = os.path.join(SCRIPT_DIR, "dummy_onnx_ref")
    os.makedirs(ref_dir, exist_ok=True)

    all_passed = True
    for batch_size in batch_sizes:
        inp, out = run_onnx_reference(MODEL_PATH, batch_size, seed=seed)
        assert inp.shape == (batch_size, INPUT_SIZE), inp.shape
        assert out.shape == (batch_size, OUTPUT_SIZE), out.shape

        np.save(os.path.join(ref_dir, f"input_b{batch_size}.npy"), inp)
        np.save(os.path.join(ref_dir, f"output_ref_b{batch_size}.npy"), out)
        print(f"  batch_size={batch_size}: input {inp.shape} -> output {out.shape}")

    print(f"Reference data saved to: {ref_dir}")
    return ref_dir


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


# Compare Triton CPU EP vs MIGraphX EP outputs (same input, both models on server)
def validate_cpu_vs_migraphx(
    url: str = "localhost:8000",
    cpu_model: str = "dummy_cpu_onnx",
    migraphx_model: str = "dummy_migraphx_onnx",
    batch_sizes=(1, 4),
    seed: int = 123,
    compare_to_ref: bool = False,
):
    """Run same input on dummy_cpu_onnx and dummy_migraphx_onnx; compare outputs."""
    rtol, atol = 1e-5, 1e-5
    base = _triton_base_url(url)
    try:
        if not _triton_is_live(base):
            print(f"ERROR: Triton server not live at {url}", file=sys.stderr)
            sys.exit(1)
        _require_models_ready(base, [cpu_model, migraphx_model])
    except Exception as e:
        print(f"ERROR: Cannot connect to Triton at {url}: {e}", file=sys.stderr)
        sys.exit(1)

    if compare_to_ref and not os.path.isfile(MODEL_PATH):
        print(f"WARNING: Model not found at {MODEL_PATH}, skipping reference comparison", file=sys.stderr)
        compare_to_ref = False

    all_passed = True
    np.random.seed(seed)
    for batch_size in batch_sizes:
        input_data = np.random.randn(batch_size, INPUT_SIZE).astype(np.float32)
        cpu_out = run_triton_inference(url, cpu_model, input_data)
        migraphx_out = run_triton_inference(url, migraphx_model, input_data)

        # CPU vs MIGraphX
        match = np.allclose(cpu_out, migraphx_out, rtol=rtol, atol=atol)
        if not match:
            diff = np.abs(cpu_out - migraphx_out)
            print(f"  batch_size={batch_size}: CPU vs MIGraphX MISMATCH (max_diff={diff.max():.6e})")
            all_passed = False
        else:
            print(f"  batch_size={batch_size}: CPU vs MIGraphX OK")

        # Optional: compare both to reference
        if compare_to_ref:
            _, ref_out = run_onnx_reference(MODEL_PATH, batch_size, seed=seed)
            for label, out in [("CPU", cpu_out), ("MIGraphX", migraphx_out)]:
                if not np.allclose(out, ref_out, rtol=rtol, atol=atol):
                    d = np.abs(out - ref_out)
                    print(f"  batch_size={batch_size}: {label} vs reference MISMATCH (max_diff={d.max():.6e})")
                    all_passed = False
                else:
                    print(f"  batch_size={batch_size}: {label} vs reference OK")

    if all_passed:
        print("CPU vs MIGraphX validation: all passed.")
    else:
        print("CPU vs MIGraphX validation: some comparisons failed.", file=sys.stderr)
        sys.exit(1)


# compare onnx reference output with tritonserver output
def validate_against_triton(url: str = "localhost:8000", model_name: str = "dummy_onnx", seed=123):
    """Compare ONNX Runtime output with Triton for the same inputs."""
    if not os.path.isfile(MODEL_PATH):
        print(f"ERROR: Model not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    ref_dir = os.path.join(SCRIPT_DIR, "dummy_onnx_ref")
    batch_sizes = (1, 4)
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

    np.random.seed(seed)
    all_passed = True
    for batch_size in batch_sizes:
        ref_inp, ref_out = run_onnx_reference(MODEL_PATH, batch_size, seed=seed)
        triton_out = run_triton_inference(url, model_name, ref_inp)
        match = np.allclose(triton_out, ref_out, rtol=rtol, atol=atol)
        if not match:
            diff = np.abs(triton_out - ref_out)
            print(f"  batch_size={batch_size}: MISMATCH (max_diff={diff.max():.6e})")
            all_passed = False
        else:
            print(f"  batch_size={batch_size}: OK (match within rtol={rtol}, atol={atol})")

    if all_passed:
        print("Triton validation: all passed.")
    else:
        print("Triton validation: some comparisons failed.", file=sys.stderr)
        sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate dummy ONNX model")
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Only run ONNX reference and save input/output .npy files",
    )
    parser.add_argument(
        "--triton",
        action="store_true",
        help="Compare ONNX reference with Triton server",
    )
    parser.add_argument(
        "--triton-url",
        default="localhost:8000",
        help="Triton server URL (default: localhost:8000)",
    )
    parser.add_argument(
        "--model-name",
        default="dummy_onnx",
        help="Triton model name (default: dummy_onnx)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Batch sizes for reference (default: 1 4 8)",
    )
    parser.add_argument(
        "--compare-cpu-migraphx",
        action="store_true",
        help="Compare Triton dummy_cpu_onnx vs dummy_migraphx_onnx (same input)",
    )
    parser.add_argument(
        "--compare-to-ref",
        action="store_true",
        help="With --compare-cpu-migraphx, also compare both to ONNX Runtime CPU reference",
    )
    args = parser.parse_args()

    if args.compare_cpu_migraphx:
        validate_cpu_vs_migraphx(
            url=args.triton_url,
            batch_sizes=tuple(args.batch_sizes),
            compare_to_ref=args.compare_to_ref,
        )
    elif args.triton:
        validate_against_triton(url=args.triton_url, model_name=args.model_name)
    else:
        validate_reference(batch_sizes=tuple(args.batch_sizes))


if __name__ == "__main__":
    main()
