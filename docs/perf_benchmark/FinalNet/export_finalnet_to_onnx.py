#!/usr/bin/env python3
"""
Export a FinalNet .model checkpoint (PyTorch state_dict) to ONNX format.

Uses model config, dataset config and the checkpoint to build feature map and export to ONNX format.

Usage:
  python export_finalnet_to_onnx.py \\
    --checkpoint /path/to/model.model \\
    --config-dir /workspace/BARS/.../FinalNet_criteo_x4_tuner_config_05 \\
    --expid FinalNet_criteo_x4_001_041_449ccb21 \\
    --output /path/to/model.onnx
"""

import argparse
import glob as _glob
import os
import sys
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FUXICTR_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
FINALNET_DIR = os.path.dirname(SCRIPT_DIR)
for p in (FUXICTR_ROOT, FINALNET_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml
import torch
from fuxictr.features import FeatureMap

try:
    from src import FinalNet
except ModuleNotFoundError as e:
    sys.exit(f"Import failed: {e}\nInstall deps: pip install pyyaml torch polars")


def _load_config(config_dir, experiment_id):
    """Load model + dataset config from config_dir."""
    model_configs = _glob.glob(os.path.join(config_dir, "model_config.yaml"))
    if not model_configs:
        model_configs = _glob.glob(os.path.join(config_dir, "model_config", "*.yaml"))
    if not model_configs:
        raise RuntimeError(f"config_dir={config_dir} is not valid!")
    found_params = {}
    for config in model_configs:
        with open(config, "r") as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if "Base" in config_dict:
                found_params["Base"] = config_dict["Base"]
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    params = {**(found_params.get("Base", {})), **found_params.get(experiment_id, {})}
    assert "dataset_id" in params, f"expid={experiment_id} is not valid in config."
    params["model_id"] = experiment_id

    dataset_id = params["dataset_id"]
    dataset_configs = _glob.glob(os.path.join(config_dir, "dataset_config.yaml"))
    if not dataset_configs:
        dataset_configs = _glob.glob(os.path.join(config_dir, "dataset_config", "*.yaml"))
    for config in dataset_configs:
        with open(config, "r") as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                params.update(config_dict[dataset_id])
                return params
    raise RuntimeError(f"dataset_id={dataset_id} is not found in config.")


def _feature_names_from_config(params):
    """Expand feature_cols from config into ordered list of feature names."""
    names = []
    for col in params.get("feature_cols", []):
        name = col.get("name")
        if isinstance(name, list):
            names.extend(name)
        else:
            names.append(name)
    return names


def _find_embedding_prefix(state_dict, feature_names):
    """Find state_dict prefix for embedding_layers (e.g. embedding_layer.embedding_layer.embedding_layers.)."""
    suffix = ".weight"
    for name in feature_names:
        for key in state_dict:
            if key.endswith(name + suffix) and "embedding_layers" in key:
                return key[: key.index(name)]
    return None


def _build_feature_map_from_config_and_checkpoint(params, state_dict):
    """
    Build FeatureMap from dataset config + checkpoint state_dict so we don't need feature_map.json.
    Infers feature types and vocab/embed dims from embedding_layer weights.
    """
    dataset_id = params["dataset_id"]
    feature_names = _feature_names_from_config(params)
    if not feature_names:
        raise RuntimeError("No feature_cols in dataset config.")

    label_col = params.get("label_col", {})
    labels = [label_col["name"]] if isinstance(label_col, dict) and "name" in label_col else []

    prefix = _find_embedding_prefix(state_dict, feature_names)
    if prefix is None:
        raise RuntimeError(
            f"Could not find embedding_layers keys in checkpoint. "
            f"Looked for state_dict keys containing a feature name (e.g. I1) and 'embedding_layers'. "
            f"Checkpoint keys sample: {list(state_dict.keys())[:5]}"
        )

    features = OrderedDict()
    for name in feature_names:
        key = prefix + name + ".weight"
        if key not in state_dict:
            raise RuntimeError(f"Checkpoint missing key {key}. Config feature order may not match the model.")
        w = state_dict[key]
        out_dim, in_dim = w.shape
        if in_dim == 1:
            features[name] = {"type": "numeric", "embedding_dim": out_dim}
        else:
            features[name] = {
                "type": "categorical",
                "vocab_size": out_dim,
                "embedding_dim": in_dim,
                "padding_idx": 0,
            }

    feature_map = FeatureMap(dataset_id, data_dir=".")
    feature_map.features = features
    feature_map.labels = labels
    feature_map.default_emb_dim = params.get("embedding_dim")
    feature_map.total_features = 0
    feature_map.set_column_index()
    feature_map.num_fields = feature_map.get_num_fields()
    return feature_map


class FinalNetONNXWrapper(torch.nn.Module):
    """
    Wraps FinalNet for ONNX: calls each embedding layer directly (no dict iteration)
    so the legacy tracer preserves shape (batch, num_fields, emb_dim); then forward1/forward2.
    """

    def __init__(self, model, feature_names):
        super().__init__()
        self.model = model
        self.feature_names = list(feature_names)
        self._emb = model.embedding_layer.embedding_layer

    def forward(self, *tensors):
        emb_layers = self._emb.embedding_layers
        emb_list = []
        for i, name in enumerate(self.feature_names):
            t = tensors[i]
            spec = self.model.feature_map.features[name]
            if spec["type"] == "numeric":
                t = t.float().view(-1, 1)
            else:
                t = t.long()
            out = emb_layers[name](t)
            emb_list.append(out)
        # (batch, 1, emb_dim) per field -> cat on dim=1 -> (batch, num_fields, emb_dim)
        feature_emb = torch.cat(emb_list, dim=1)
        if self.model.block_type == "1B":
            y_pred = self.model.forward1(feature_emb)
        else:
            y1 = self.model.forward1(feature_emb)
            y2 = self.model.forward2(feature_emb)
            y_pred = 0.5 * (y1 + y2)
        return self.model.output_activation(y_pred)


def _example_inputs(feature_map, feature_names, batch_size=4, device="cpu"):
    """Build example tensors for ONNX export from feature_map."""
    example_inputs = []
    for name in feature_names:
        spec = feature_map.features[name]
        feat_type = spec.get("type", "categorical")
        if feat_type == "numeric":
            t = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
        else:
            vocab_size = max(spec.get("vocab_size", 2), 2)
            t = torch.randint(0, vocab_size, (batch_size, 1), dtype=torch.long, device=device)
        example_inputs.append(t)
    return tuple(example_inputs)


def main():
    parser = argparse.ArgumentParser(description="Export FinalNet .model checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .model file")
    _default_config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(FUXICTR_ROOT))),
        "BARS", "ranking", "ctr", "FinalNet", "FinalNet_criteo_x4_001", "FinalNet_criteo_x4_tuner_config_05",
    )
    if not os.path.isdir(_default_config):
        _default_config = os.path.join(FUXICTR_ROOT, "config")
    parser.add_argument("--config-dir", type=str, default=_default_config,
                        help="Directory with model_config.yaml and dataset_config.yaml")
    parser.add_argument("--expid", type=str, default="FinalNet_criteo_x4_001_041_449ccb21",
                        help="Experiment ID in model_config.yaml")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .onnx path. Default: checkpoint path with .onnx")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for ONNX dynamic_axes")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        sys.exit(f"Checkpoint not found: {checkpoint_path}")

    config_dir = os.path.abspath(args.config_dir)
    if not os.path.isdir(config_dir):
        sys.exit(f"Config dir not found: {config_dir}")

    params = _load_config(config_dir, args.expid)
    params["gpu"] = -1

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    feature_map = _build_feature_map_from_config_and_checkpoint(params, state_dict)

    feature_names = [f for f in feature_map.features.keys() if f not in feature_map.labels]
    num_fields = len(feature_names)

    model = FinalNet(feature_map, **params)
    model.load_weights(checkpoint_path)
    model.eval()

    wrapper = FinalNetONNXWrapper(model, feature_names)
    wrapper.eval()
    example_inputs = _example_inputs(feature_map, feature_names, batch_size=args.batch_size)

    out_path = args.output
    if out_path is None:
        out_path = os.path.splitext(checkpoint_path)[0] + ".onnx"
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    input_names = [f"input_{i}" for i in range(num_fields)]
    # Dynamic batch dimension (axis 0) named "b" in ONNX
    dynamic_axes = {"y_pred": {0: "b"}}
    for i in range(num_fields):
        dynamic_axes[input_names[i]] = {0: "b"}

    torch.onnx.export(
        wrapper,
        example_inputs,
        out_path,
        input_names=input_names,
        output_names=["y_pred"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"Exported to {out_path}")


if __name__ == "__main__":
    main()
