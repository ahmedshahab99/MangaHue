from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .contract import CONTRACT_ID, canonical_hash, sha256_file
from .dataset import MangaHueDataset
from .models import build_model


def _metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = pred - target
    mse = float(error.square().mean())
    mae = float(error.abs().mean())
    psnr = 99.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)
    weights = pred.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    pred_luma, target_luma = (pred * weights).sum(1), (target * weights).sum(1)
    luma_mae = float((pred_luma - target_luma).abs().mean())
    pred_dx, target_dx = pred[..., :, 1:] - pred[..., :, :-1], target[..., :, 1:] - target[..., :, :-1]
    pred_dy, target_dy = pred[..., 1:, :] - pred[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    edge_mae = float(((pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean()) / 2)
    return {"mae": mae, "mse": mse, "psnr": psnr, "luminance_mae": luma_mae, "edge_mae": edge_mae}


def evaluate(checkpoint: Path, manifest: Path, size: int, batch_size: int, device: str) -> dict[str, object]:
    dataset = MangaHueDataset(manifest, size=size, synthetic_hints=False)
    if not dataset.samples:
        raise ValueError("evaluation manifest is empty")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()
    totals = {key: 0.0 for key in ("mae", "mse", "psnr", "luminance_mae", "edge_mae")}
    batches = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            values = _metrics(model(inputs.to(device)), targets.to(device))
            for key, value in values.items():
                totals[key] += value
            batches += 1
    metrics = {key: value / batches for key, value in totals.items()}
    result: dict[str, object] = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest_sha256": sha256_file(manifest),
        "samples": len(dataset),
        "metrics": metrics,
    }
    result["evaluation_sha256"] = canonical_hash(result)
    return result


def enforce_regression_gate(current: dict[str, object], baseline: dict[str, object], tolerance: float) -> None:
    current_metrics = current["metrics"]
    baseline_metrics = baseline["metrics"]
    assert isinstance(current_metrics, dict) and isinstance(baseline_metrics, dict)
    failures: list[str] = []
    for metric in ("mae", "mse", "luminance_mae", "edge_mae"):
        allowed = float(baseline_metrics[metric]) * (1.0 + tolerance)
        if float(current_metrics[metric]) > allowed:
            failures.append(f"{metric}={current_metrics[metric]:.6f} exceeds {allowed:.6f}")
    minimum_psnr = float(baseline_metrics["psnr"]) * (1.0 - tolerance)
    if float(current_metrics["psnr"]) < minimum_psnr:
        failures.append(f"psnr={current_metrics['psnr']:.4f} is below {minimum_psnr:.4f}")
    if failures:
        raise ValueError("checkpoint regression gate failed: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and regression-gate a MangaHue checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = evaluate(args.checkpoint, args.manifest, args.size, args.batch_size, args.device)
    if args.baseline:
        enforce_regression_gate(result, json.loads(args.baseline.read_text(encoding="utf-8")), args.tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"evaluation gate passed -> {args.output}")


if __name__ == "__main__":
    main()
