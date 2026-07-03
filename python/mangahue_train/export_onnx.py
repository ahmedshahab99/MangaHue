from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MangaHue Hint U-Net to ONNX")
    parser.add_argument("--checkpoint", type=Path, help="Optional PyTorch checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=768)
    parser.add_argument("--input-channels", type=int, default=5)
    parser.add_argument("--validate", action="store_true", help="Run validate_onnx against the exported model after writing it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(input_channels=args.input_channels)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    dummy = torch.zeros(1, args.input_channels, args.input_size, args.input_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["image_input"],
        output_names=["color_output"],
        dynamic_axes={
            "image_input": {2: "height", 3: "width"},
            "color_output": {2: "height", 3: "width"},
        },
        opset_version=17,
    )

    metadata = {
        "model_name": "MangaHue Hint U-Net",
        "version": "0.1.0",
        "model_type": "hint_unet",
        "input_size": args.input_size,
        "input_channels": args.input_channels,
        "output_channels": 3,
        "input_range": "0_1",
        "output_range": "0_1",
        "expects_lineart_inverted": False,
        "supports_style_vector": False,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"exported {args.output}")
    print(f"wrote {args.metadata}")

    if args.validate:
        from .validate_onnx import main as run_validate
        import sys

        old_argv = sys.argv
        try:
            sys.argv = [
                "validate_onnx",
                "--model",
                str(args.output),
                "--metadata",
                str(args.metadata),
            ]
            run_validate()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    main()

