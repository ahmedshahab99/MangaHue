"""Export the reference-CGAN checkpoint (074000_gray.pt) to a single ONNX graph.

Runs ONCE on the production VPS (where torch is installed) so the Rust engine
can run the model natively through the `ort` crate with no Python at runtime.

Architecture (mirrors python/mangahue_train/reference_cgan_worker.py):
    ReferenceEncoder (VGG19 features through layer 31 + feature2vector) embeds
    a 3-channel sRGB image into a (1, 512, 1, 1) style vector; ColorUNet maps
    the CIELAB L channel normalized as (L - 50) / 50 plus the style vector to
    the ab channels, passed through a final Tanh. See worker lines 165-196
    (encoder) and 131-162 (ColorUNet). Both classes are imported verbatim from
    reference_cgan_worker.py; the checkpoint state-dict key juggling also
    mirrors the worker exactly (worker lines 218-236).

SINGLE-INPUT COMPROMISE (read before wiring the Rust backend):
    The worker's real runtime contract has TWO inputs: a separate reference
    style image, embedded once per worker (worker lines 241-252), and the
    lineart lightness fed to ColorUNet (worker lines 255-269). A single-input
    ONNX graph cannot reproduce that, so the exported graph derives BOTH the
    style embedding and the lightness from the ONE input tensor. The encoder
    is therefore no longer run once per worker: it runs inside the graph on
    every forward pass.

INPUT contract ("input"):
    float32 (or float16 with --fp16) tensor, NCHW, shape (1, 3, size, size),
    sRGB values in [0, 1]. The engine must feed the grayscale lineart
    replicated to 3 channels (all three channels identical), already resized
    to size x size — this is the worker's `Image.open(lineart_path).convert(
    "RGB")` then `/ 255.0` (worker lines 256-257). The graph applies the
    worker's normalizations itself:
      * encoder path: (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        (worker lines 187-190), after a bilinear resize (worker lines 245-252);
      * colorizer path: CIELAB L* (skimage rgb2lab, D65/2 observer) then
        (L - 50) / 50 (worker lines 258-261), after a bilinear resize
        (worker lines 263-268).
    Both bilinear resizes (mode="bilinear", align_corners=False) are identity
    at size x size but are kept so the graph matches the worker op-for-op.

OUTPUT contract ("output"):
    float32/float16 tensor, NCHW, shape (1, 2, size, size). Channel 0 =
    CIELAB a, channel 1 = CIELAB b, tanh range [-1, 1] (ColorUNet's final
    Tanh, worker lines 143-148). Exact worker post-processing to final sRGB
    (worker lines 270-288):
        ab_lab = output * 110.0                      # worker line 271
        lab    = concat(L_lineart, a, b)             # worker lines 276-282
        rgb    = clip(lab2rgb(lab), 0, 1) * 255      # worker lines 283-287
        (then cast to uint8)
    The engine must supply L_lineart itself (CIELAB L of the lineart via its
    own rgb2lab, normalized identically) because the graph only outputs ab.

Validation gates (exit 0 only if all pass):
    * onnx.checker.check_model
    * onnxruntime (CPU) vs torch on the same dummy input:
      max abs diff < 1e-3 (fp32) or < 1e-2 (fp16)
    * in-graph L* vs skimage color.rgb2lab parity
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as functional

from .reference_cgan_worker import ColorUNet, ReferenceEncoder


def _lab_lightness(rgb: torch.Tensor) -> torch.Tensor:
    """CIELAB L* (D65/2 deg) of sRGB in [0, 1], mirroring skimage rgb2lab.

    Only the Y channel of XYZ is needed for L*: L = 116 * f(y / yn) - 16 with
    yn = 1.0. Constants are the exact skimage colorconv values (xyz_from_rgb
    matrix row for Y, (6/29)**3 knee, 3*(6/29)**2 slope, 4/29 offset) and the
    output is already normalized as the worker does: (L - 50) / 50.
    """
    dtype = rgb.dtype
    linear = torch.where(
        rgb <= torch.tensor(0.04045, dtype=dtype),
        rgb / torch.tensor(12.92, dtype=dtype),
        ((rgb + torch.tensor(0.055, dtype=dtype)) / torch.tensor(1.055, dtype=dtype))
        ** torch.tensor(2.4, dtype=dtype),
    )
    y = (
        linear[:, 0:1, :, :] * torch.tensor(0.212671, dtype=dtype)
        + linear[:, 1:2, :, :] * torch.tensor(0.715160, dtype=dtype)
        + linear[:, 2:3, :, :] * torch.tensor(0.072169, dtype=dtype)
    )
    epsilon = torch.tensor((6.0 / 29.0) ** 3, dtype=dtype)
    kappa = torch.tensor(3.0 * (6.0 / 29.0) ** 2, dtype=dtype)
    offset = torch.tensor(4.0 / 29.0, dtype=dtype)
    fy = torch.where(
        y > epsilon,
        torch.pow(y, torch.tensor(1.0 / 3.0, dtype=dtype)),
        y / kappa + offset,
    )
    return (116.0 * fy - torch.tensor(66.0, dtype=dtype)) / torch.tensor(
        50.0, dtype=dtype
    )


class _ReferenceCganPipeline(nn.Module):
    """Single-input composition of encoder + ColorUNet, for ONNX export.

    The single input stands in for both the worker's reference image (style
    embedding) and its lineart (lightness), as documented in the module
    docstring.
    """

    def __init__(self, encoder: ReferenceEncoder, colorizer: ColorUNet, size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.colorizer = colorizer
        self.size = size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Worker reference path (lines 245-252): bilinear resize, then encode.
        resized = functional.interpolate(
            image, size=(self.size, self.size), mode="bilinear", align_corners=False
        )
        style = self.encoder(resized)
        # Worker lineart path (lines 258-268): L* -> (L - 50) / 50 -> resize.
        lightness = _lab_lightness(image)
        lightness = functional.interpolate(
            lightness, size=(self.size, self.size), mode="bilinear", align_corners=False
        )
        return self.colorizer(lightness, style)


def load_checkpoint(path: Path) -> dict:
    """Load the checkpoint, preferring weights_only=True like the worker."""
    try:
        with torch.serialization.safe_globals([argparse.Namespace]):
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        warnings.warn(
            f"weights_only=True failed ({error}); retrying with weights_only=False",
            stacklevel=2,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def build_pipeline(checkpoint: dict, size: int) -> _ReferenceCganPipeline:
    """Instantiate encoder + ColorUNet and load state, mirroring the worker."""
    encoder_state = checkpoint["colorEncoder"]
    encoder = ReferenceEncoder()
    vgg_prefix = "vgg.vgg_model.features."
    vgg_state = {
        key[len(vgg_prefix) :]: value
        for key, value in encoder_state.items()
        if key.startswith(vgg_prefix)
    }
    vector_prefix = "feature2vector."
    vector_state = {
        key[len(vector_prefix) :]: value
        for key, value in encoder_state.items()
        if key.startswith(vector_prefix)
    }
    encoder.vgg_features.load_state_dict(vgg_state)
    encoder.feature2vector.load_state_dict(vector_state)
    colorizer = ColorUNet()
    colorizer.load_state_dict(checkpoint["colorUNet"])
    return _ReferenceCganPipeline(encoder, colorizer, size)


def freeze(model: nn.Module) -> None:
    """Eval mode with all parameters frozen, for ONNX export."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the reference-CGAN checkpoint to a fixed-size single-input ONNX graph"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to 074000_gray.pt"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .onnx path")
    parser.add_argument(
        "--size", type=int, default=256, help="Model input size (fixed, default 256)"
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Cast weights and graph to float16"
    )
    parser.add_argument(
        "--opset", type=int, default=17, help="ONNX opset version (default 17)"
    )
    return parser.parse_args()


def _verify_lightness_parity(args: argparse.Namespace, dummy: torch.Tensor) -> None:
    """Gate: the in-graph L* preprocessing must match skimage rgb2lab."""
    try:
        from skimage import color
    except ImportError:
        warnings.warn("skimage not available; skipping L* parity check", stacklevel=2)
        return
    rgb = dummy.detach().cpu().numpy().astype(np.float32)
    lab = color.rgb2lab(rgb[0].transpose(1, 2, 0))  # (H, W, 3) float64
    expected = ((lab[..., 0] - 50.0) / 50.0).astype(np.float32)
    graph_lightness = (
        _lab_lightness(dummy).detach().cpu().numpy()[0, 0].astype(np.float32)
    )
    diff = float(np.max(np.abs(graph_lightness - expected)))
    limit = 0.25 if args.fp16 else 1e-2
    if diff >= limit:
        raise RuntimeError(
            f"in-graph L* deviates from skimage rgb2lab by {diff:.3e} "
            f"(limit {limit:.1e})"
        )
    print(f"in-graph L* vs skimage rgb2lab max abs diff: {diff:.3e} (limit {limit:.1e}) OK")


def verify(args: argparse.Namespace, pipeline: nn.Module, dummy: torch.Tensor) -> None:
    """Validation gates; raises RuntimeError when any gate fails."""
    print(f"validating {args.output}")
    model = onnx.load(str(args.output))
    onnx.checker.check_model(model)
    print("onnx.checker: OK")
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    onnx_output = np.asarray(
        session.run(["output"], {"input": dummy.numpy()})[0], dtype=np.float32
    )
    torch_output = np.asarray(pipeline(dummy).detach().numpy(), dtype=np.float32)
    expected_shape = (1, 2, args.size, args.size)
    if onnx_output.shape != expected_shape:
        raise RuntimeError(
            f"unexpected ONNX output shape {onnx_output.shape}; expected {expected_shape}"
        )
    diff = float(np.max(np.abs(onnx_output - torch_output)))
    limit = 1e-2 if args.fp16 else 1e-3
    if diff >= limit:
        raise RuntimeError(
            f"onnxruntime vs torch max abs diff {diff:.3e} exceeds {limit:.1e}"
        )
    print(f"onnxruntime (CPU) vs torch max abs diff: {diff:.3e} (limit {limit:.1e}) OK")
    _verify_lightness_parity(args, dummy)
    size_mb = args.output.stat().st_size / 1e6
    print(
        f"input tensor: name=input shape={tuple(dummy.shape)} dtype={dummy.dtype} "
        f"sRGB [0, 1] NCHW"
    )
    print(
        f"output tensor: name=output shape={onnx_output.shape} "
        f"channels=[a, b] range=[{onnx_output.min():.4f}, {onnx_output.max():.4f}] (tanh)"
    )
    print(f"model size: {size_mb:.1f} MB")


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    print(f"loading checkpoint {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)
    pipeline = build_pipeline(checkpoint, args.size)
    del checkpoint
    freeze(pipeline)
    dtype_name = "float16" if args.fp16 else "float32"
    if args.fp16:
        pipeline = pipeline.half()
    # Grayscale replicated to 3 channels, like the worker's lineart path.
    dummy = torch.rand(1, 1, args.size, args.size).expand(
        1, 3, args.size, args.size
    ).contiguous()
    if args.fp16:
        dummy = dummy.half()
    print(f"tracing pipeline (dtype={dtype_name}, size={args.size})")
    traced = torch.jit.trace(pipeline, dummy)
    print(f"exporting ONNX (opset {args.opset}) to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        traced,
        dummy,
        str(args.output),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        opset_version=args.opset,
    )
    verify(args, pipeline, dummy)


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    print("ALL GATES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
