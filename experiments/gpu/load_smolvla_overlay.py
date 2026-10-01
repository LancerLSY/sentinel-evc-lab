"""Load the evaluated SmolVLA overlay and predict one offline action chunk.

Requires the pinned LeRobot 0.6.1 experiment environment and CUDA. This is an
offline loading example; its output is never sent to a robot controller.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import train_smolvla as trainer


def load_policy(bundle: Path, base: Path, vlm: Path):
    torch, config_class, _, make_policy, processors, _ = trainer._require_dependencies()
    identity = json.loads((bundle / "run_identity.json").read_text())
    checkpoint = json.loads((bundle / "checkpoint.json").read_text())
    stats = json.loads((bundle / "train_only_stats.json").read_text())
    info_path = bundle / "dataset_info.json"
    info = json.loads(info_path.read_text())
    if trainer._sha256_file(Path(trainer.__file__)) != identity["bindings"]["script_sha256"]:
        raise ValueError("Use the training source recorded in this bundle")
    if trainer._sha256_value(identity["bindings"]) != identity["run_identity"] or checkpoint["run_identity"] != identity["run_identity"]:
        raise ValueError("Run identity does not match its frozen bindings")
    if stats["fit_split"] != "train" or trainer._sha256_bytes(trainer._json_bytes(stats["stats"])) != checkpoint["train_stats_sha256"]:
        raise ValueError("Normalization does not match the evaluated checkpoint")
    for name, directory in [("base_model", base), ("vlm_backbone", vlm)]:
        for item in identity["bindings"]["downloads"][name]["files"]:
            file = directory / item["path"]
            if file.stat().st_size != item["bytes"] or trainer._sha256_file(file) != item["sha256"]:
                raise ValueError(f"Pinned upstream file changed: {name}/{item['path']}")
    dataset_info = next(x for x in identity["bindings"]["downloads"]["dataset"]["files"] if x["path"] == "meta/info.json")
    if trainer._sha256_file(info_path) != dataset_info["sha256"]:
        raise ValueError("Dataset feature metadata changed")
    weight_path = bundle / "trainable_state.pt"
    if trainer._sha256_file(weight_path) != checkpoint["trainable_state_sha256"]:
        raise ValueError("Overlay bytes differ from the evaluated checkpoint")
    tensor_stats = {key: {name: torch.tensor(value, dtype=torch.float32) for name, value in fields.items()} for key, fields in stats["stats"].items()}
    camera_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    metadata = SimpleNamespace(features=info["features"], camera_keys=camera_keys, stats=tensor_stats)
    args = SimpleNamespace(model_path=base, vlm_path=vlm, device="cuda")
    policy, preprocessor, postprocessor, config, names = trainer._build_policy(config_class, make_policy, processors, args, metadata, tensor_stats, torch)
    weights = torch.load(weight_path, map_location="cpu", weights_only=True)
    if set(weights) != set(names) or not all(torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("Overlay tensor set or values changed")
    trainer._load_trainable_state(policy, weights)
    policy.eval()
    return policy, preprocessor, postprocessor, camera_keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, required=True)
    parser.add_argument("--example", type=Path, required=True, help="Saved dev-only RGB/state/task input; not actuator commands")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    policy, preprocessor, postprocessor, cameras = load_policy(args.bundle, args.base, args.vlm)
    example = torch.load(args.example, map_location="cpu", weights_only=True)
    batch = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in example["batch"].items()}
    prepared = trainer._to_model_batch(batch, cameras, preprocessor, torch)
    trainer._validate_preprocessed_batch(prepared, policy.config, cameras, torch)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = trainer._dataset_native_action(postprocessor, policy.predict_action_chunk(prepared, noise=example["noise"].to("cuda"))).float().cpu()
    if tuple(output.shape) != (1, 50, 6) or not torch.isfinite(output).all():
        raise ValueError("Unexpected action chunk")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"action_chunk": output, "scope": "offline dataset-native values; physical units undeclared; no robot commanded"}, args.out)
    print(json.dumps({"shape": list(output.shape), "finite": True, "output": str(args.out)}))


if __name__ == "__main__":
    main()
