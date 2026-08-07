"""Export a validated int8 ONNX quantization of a MangaHue colorizer model.

The engine runs ONNX models natively via the Rust `ort` crate, so int8
quantization happens offline in Python. A previous attempt
(manga-colorize-int8.onnx) produced a corrupt artifact because
DynamicQuantizeLinear was applied directly to an fp16 graph. This script
refuses to repeat that: it converts fp16 models to fp32 first, quantizes with
onnxruntime, and only exits 0 after every validation gate passes.

Pipeline:
    1. Load --model with onnx. If any float16 initializer, Constant-style
       node tensor attribute, or graph input/output type is found, cast it to
       float32 with onnx.numpy_helper (the tensor data_type proto is
       rewritten, so Constant nodes are fixed in place) and verify the result
       with onnx.checker. The cast is lossless: every fp16 value is exactly
       representable in fp32.
    2. Quantize with onnxruntime.quantization:
       - static (default): quantize_static with a CalibrationDataReader that
         feeds the --calibration-images resized to --spatial, normalized to
         0..1, NCHW float32. per_channel=True, QInt8 activations and weights
         with ActivationSymmetric; if the QInt8 activation pass fails, it
         retries with QUInt8 activations.
       - dynamic (--dynamic): quantize_dynamic with QInt8 weights; no
         calibration data is needed.
    3. Validation gates (any failure -> exit 1 with the failing gate named):
       - onnx.checker.check_model on the int8 artifact;
       - both fp32 and int8 load in onnxruntime InferenceSession
         (CPUExecutionProvider);
       - input/output names and shapes match the fp32 model (first input /
         first output);
       - numeric diff on zeros, ones, and a seeded-random input: max abs
         diff < --tolerance (default 0.05 on 0..1 outputs) and mean abs
         diff < tolerance / 4.

Requirements:
    pip install onnx onnxruntime onnxconverter-common pillow numpy
    (onnxconverter-common is optional; the built-in fp16->fp32 converter uses
    onnx.numpy_helper and onnx.checker instead)

Usage examples (run from the repo root or from python/):
    # manga-colorize (hint U-Net, 5 channels, dynamic spatial), static
    python -m mangahue_train.export_quantized \
        --model models/manga-colorize-fp16.onnx \
        --output models/manga-colorize-int8.onnx \
        --calibration-images test_images --channels 5 --spatial 768

    # ddcolor-tiny (3 channels, 512x512), dynamic (no calibration data)
    python -m mangahue_train.export_quantized \
        --model models/ddcolor-tiny-fp16.onnx \
        --output models/ddcolor-tiny-int8.onnx --dynamic

Caveats:
    - Static quantization needs representative calibration images. For the
      5-channel hint contract the reader feeds [gray, r, g, b, mask=1].
    - Ops the quantizer does not support (e.g. some Resize/Cast patterns or
      exotic ops) are left in fp32 or abort the run; the QInt8 -> QUInt8
      activation fallback and the numeric gate keep silent corruption from
      being shipped.
    - The int8 artifact is standard QDQ-format ONNX and loads in the Rust
      ort crate exactly like the fp16 model, but only deploy it if the
      script exits 0 (all gates passed).
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, checker, numpy_helper, shape_inference
from onnxruntime.quantization import QuantType, quantize_dynamic, quantize_static
from onnxruntime.quantization.calibration import CalibrationDataReader
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _print_table(rows: list[tuple[str, str]]) -> None:
    width = max(len(label) for label, _ in rows) + 2
    for label, value in rows:
        print(f"{label:<{width}}{value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a MangaHue ONNX colorizer to int8 (static QDQ or dynamic) and validate the artifact"
    )
    parser.add_argument("--model", type=Path, required=True, help="fp32 ONNX model to quantize (fp16 models are converted to fp32 automatically)")
    parser.add_argument("--output", type=Path, required=True, help="output int8 ONNX path")
    parser.add_argument(
        "--calibration-images",
        type=str,
        default=None,
        help="directory or comma-separated image paths used to calibrate static quantization (required unless --dynamic)",
    )
    parser.add_argument("--dynamic", action="store_true", help="use dynamic quantization (no calibration data needed)")
    parser.add_argument("--channels", type=int, default=3, help="input channels of the model (default 3)")
    parser.add_argument("--spatial", type=int, default=512, help="input spatial size of the model (default 512)")
    parser.add_argument("--batch", type=int, default=1, help="batch size used for calibration and validation (default 1)")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="max absolute output diff allowed on 0..1 outputs (default 0.05); mean diff must stay below tolerance / 4",
    )
    return parser.parse_args()


def resolve_calibration_images(spec: str | None) -> list[Path]:
    """Expand a directory or comma-separated list of image paths."""
    if not spec:
        return []
    images: list[Path] = []
    missing: list[str] = []
    for part in [item.strip() for item in spec.split(",") if item.strip()]:
        path = Path(part)
        if path.is_dir():
            images.extend(sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file():
            images.append(path)
        else:
            missing.append(part)
    if missing:
        raise SystemExit("calibration image(s) not found: " + ", ".join(missing))
    return images


def has_fp16(model: onnx.ModelProto) -> bool:
    """True if any float16 tensor lives in the graph (initializers, node tensor attributes, or I/O types)."""
    graph = model.graph
    if any(initializer.data_type == TensorProto.FLOAT16 for initializer in graph.initializer):
        return True
    if any(
        attribute.type == AttributeProto.TENSOR and attribute.t.data_type == TensorProto.FLOAT16
        for node in graph.node
        for attribute in node.attribute
    ):
        return True
    if any(
        value.type.tensor_type.elem_type == TensorProto.FLOAT16
        for value in list(graph.input) + list(graph.output) + list(graph.value_info)
    ):
        return True
    return False


def to_fp32(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    """Cast every float16 tensor in the graph to float32 in place.

    Handles initializers, Constant-style node tensor attributes (the tensor
    data_type proto is rewritten, which is exactly what "fix Constant nodes"
    requires), graph input/output types, and value_info. Returns the model
    and the number of converted tensors. The cast is lossless.
    """
    graph = model.graph
    count = 0
    for initializer in graph.initializer:
        if initializer.data_type == TensorProto.FLOAT16:
            array = numpy_helper.to_array(initializer).astype(np.float32)
            initializer.CopyFrom(numpy_helper.from_array(array, initializer.name))
            count += 1
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == AttributeProto.TENSOR and attribute.t.data_type == TensorProto.FLOAT16:
                array = numpy_helper.to_array(attribute.t).astype(np.float32)
                attribute.t.CopyFrom(numpy_helper.from_array(array, attribute.t.name))
                count += 1
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type == TensorProto.FLOAT16:
            tensor_type.elem_type = TensorProto.FLOAT
            count += 1
    try:
        # Rebuild value_info so no stale fp16 metadata survives.
        return shape_inference.infer_shapes(model), count
    except Exception as exc:
        print(f"warning: shape inference after fp16->fp32 conversion failed: {exc}")
        return model, count


@contextmanager
def _temporary_fp32_path(model: onnx.ModelProto) -> Generator[None, None, str]:
    """Write the in-memory fp32 model to a temp file for the ORT quantizers."""
    fd, raw_path = tempfile.mkstemp(prefix="mangahue-quant-", suffix=".onnx")
    os.close(fd)
    path = Path(raw_path)
    try:
        onnx.save(model, str(path))
        yield str(path)
    finally:
        try:
            path.unlink()
        except OSError:
            pass


class ImageCalibrationReader(CalibrationDataReader):
    """Feeds calibration images to quantize_static as NCHW float32 batches in 0..1."""

    def __init__(self, images: list[Path], input_name: str, channels: int, spatial: int, batch: int) -> None:
        self._images = images
        self._input_name = input_name
        self._channels = channels
        self._spatial = spatial
        self._batch = batch
        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self._images):
            return None
        chunk = self._images[self._index : self._index + self._batch]
        self._index += self._batch
        batch_data = np.stack([self._preprocess(path) for path in chunk], axis=0)
        return {self._input_name: batch_data}

    def rewind(self) -> None:
        self._index = 0

    def _preprocess(self, path: Path) -> np.ndarray:
        try:
            image = Image.open(path)
        except OSError as exc:
            raise ValueError(f"cannot open calibration image {path}: {exc}") from exc
        image = image.resize((self._spatial, self._spatial), Image.Resampling.LANCZOS)
        if self._channels == 1:
            array = np.asarray(image.convert("L"), dtype=np.float32)[np.newaxis, :, :]
        elif self._channels == 3:
            array = np.transpose(np.asarray(image.convert("RGB"), dtype=np.float32), (2, 0, 1))
        elif self._channels == 5:
            # MangaHue hint contract: [lineart, hint_r, hint_g, hint_b, mask].
            gray = np.asarray(image.convert("L"), dtype=np.float32)[np.newaxis, :, :]
            rgb = np.transpose(np.asarray(image.convert("RGB"), dtype=np.float32), (2, 0, 1))
            array = np.concatenate([gray, rgb, np.ones_like(gray)], axis=0)
        else:
            raise ValueError(f"unsupported --channels {self._channels} (supported: 1, 3, 5)")
        return array / 255.0


def _quantize_static(model: onnx.ModelProto, output_path: Path, reader: CalibrationDataReader) -> str:
    with _temporary_fp32_path(model) as fp32_path:
        try:
            print("static quantization: QInt8 activations, QInt8 weights, per-channel")
            quantize_static(
                fp32_path,
                str(output_path),
                reader,
                per_channel=True,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                extra_options={"ActivationSymmetric": True},
            )
            return "QInt8 static (per-channel, symmetric activations)"
        except Exception as exc:
            print(f"warning: static QInt8 quantization failed ({exc}); retrying with QUInt8 activations")
            try:
                quantize_static(
                    fp32_path,
                    str(output_path),
                    reader,
                    per_channel=True,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    extra_options={"ActivationSymmetric": False},
                )
                return "QUInt8/QInt8 static (per-channel, activation fallback)"
            except Exception as exc2:
                raise SystemExit(f"static quantization failed: {exc2}") from exc2


def _quantize_dynamic(model: onnx.ModelProto, output_path: Path) -> str:
    with _temporary_fp32_path(model) as fp32_path:
        try:
            print("dynamic quantization: QInt8 weights")
            quantize_dynamic(fp32_path, str(output_path), weight_type=QuantType.QInt8)
            return "QInt8 dynamic"
        except Exception as exc:
            raise SystemExit(f"dynamic quantization failed: {exc}") from exc


def quantize_model(
    model: onnx.ModelProto,
    output_path: Path,
    reader: CalibrationDataReader | None,
    dynamic: bool,
) -> str:
    """Quantize to int8 and return a human-readable quant type label."""
    if dynamic:
        return _quantize_dynamic(model, output_path)
    if reader is None:
        raise SystemExit("internal error: static quantization requires a calibration reader")
    return _quantize_static(model, output_path, reader)


def _shape_compatible(a: list[Any], b: list[Any]) -> bool:
    """True if two ORT shape lists agree, treating dynamic dims as wildcards."""
    if len(a) != len(b):
        return False
    for dim_a, dim_b in zip(a, b):
        normalized_a = None if dim_a is None or dim_a == 0 or isinstance(dim_a, str) else dim_a
        normalized_b = None if dim_b is None or dim_b == 0 or isinstance(dim_b, str) else dim_b
        if normalized_a is not None and normalized_b is not None and normalized_a != normalized_b:
            return False
    return True


def _diff_stats(reference: np.ndarray, quantized: np.ndarray) -> tuple[float, float]:
    delta = np.abs(reference.astype(np.float64) - quantized.astype(np.float64))
    return float(np.max(delta)), float(np.mean(delta))


def validate_output(
    output_path: Path,
    fp32_bytes: bytes,
    channels: int,
    spatial: int,
    batch: int,
    tolerance: float,
) -> tuple[bool, list[tuple[str, str]]]:
    """Run every validation gate; returns (all_passed, rows of label/PASS-or-FAIL)."""
    rows: list[tuple[str, str]] = []

    try:
        checker.check_model(onnx.load(str(output_path)))
        rows.append(("onnx.checker", "PASS: artifact is structurally valid"))
    except Exception as exc:
        rows.append(("onnx.checker", f"FAIL: {exc}"))

    try:
        session_fp32 = ort.InferenceSession(fp32_bytes, providers=["CPUExecutionProvider"])
        session_int8 = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        rows.append(("session load", f"FAIL: {exc}"))
        return False, rows
    rows.append(("session load", "PASS: fp32 and int8 both load on CPU"))

    in_fp32 = session_fp32.get_inputs()[0]
    out_fp32 = session_fp32.get_outputs()[0]
    in_int8 = session_int8.get_inputs()[0]
    out_int8 = session_int8.get_outputs()[0]

    if in_fp32.name == in_int8.name and out_fp32.name == out_int8.name:
        rows.append(("io names", f"PASS: {in_int8.name} -> {out_int8.name}"))
    else:
        rows.append(("io names", f"FAIL: fp32 {in_fp32.name}->{out_fp32.name}, int8 {in_int8.name}->{out_int8.name}"))

    if _shape_compatible(in_fp32.shape, in_int8.shape) and _shape_compatible(out_fp32.shape, out_int8.shape):
        rows.append(("io shapes", f"PASS: in {in_fp32.shape} -> out {out_fp32.shape}"))
    else:
        rows.append(
            ("io shapes", f"FAIL: fp32 in {in_fp32.shape} out {out_fp32.shape}; int8 in {in_int8.shape} out {out_int8.shape}")
        )

    shape = (batch, channels, spatial, spatial)
    rng = np.random.default_rng(0)
    synthetic: list[tuple[str, np.ndarray]] = [
        ("zeros", np.zeros(shape, dtype=np.float32)),
        ("ones", np.ones(shape, dtype=np.float32)),
        ("random", rng.random(shape).astype(np.float32)),
    ]
    for label, data in synthetic:
        try:
            reference = session_fp32.run([out_fp32.name], {in_fp32.name: data})[0]
            quantized = session_int8.run([out_int8.name], {in_int8.name: data})[0]
            if reference.shape != quantized.shape:
                rows.append((f"numeric diff {label}", f"FAIL: output shape {reference.shape} != {quantized.shape}"))
                continue
            max_abs, mean_abs = _diff_stats(reference, quantized)
            if max_abs < tolerance and mean_abs < tolerance / 4:
                rows.append(
                    (
                        f"numeric diff {label}",
                        f"PASS: max_abs={max_abs:.4f} < {tolerance}, mean_abs={mean_abs:.4f} < {tolerance / 4:.4f}",
                    )
                )
            else:
                rows.append(
                    (
                        f"numeric diff {label}",
                        f"FAIL: max_abs={max_abs:.4f} >= {tolerance} or mean_abs={mean_abs:.4f} >= {tolerance / 4:.4f}",
                    )
                )
        except Exception as exc:
            rows.append((f"numeric diff {label}", f"FAIL: {exc}"))

    return all(value.startswith("PASS") for _, value in rows), rows


def main() -> None:
    args = parse_args()

    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")
    if args.spatial < 1:
        raise SystemExit("--spatial must be >= 1")
    if args.channels not in (1, 3, 5):
        raise SystemExit("--channels must be 1, 3, or 5 (calibration layout is only defined for those)")
    if args.tolerance <= 0:
        raise SystemExit("--tolerance must be > 0")

    print(f"loading {args.model}")
    try:
        model = onnx.load(str(args.model))
    except Exception as exc:
        raise SystemExit(f"failed to load {args.model}: {exc}") from exc

    converted = 0
    if has_fp16(model):
        print("model contains float16 tensors: converting to fp32 before quantization (int8 quantization of an fp16 graph is invalid)")
        model, converted = to_fp32(model)
        print(f"converted {converted} float16 tensor(s) to float32")
    else:
        print("model is already fp32")
    try:
        checker.check_model(model)
    except Exception as exc:
        raise SystemExit(f"input model fails onnx.checker: {exc}") from exc
    print("onnx.checker PASS on the fp32 model")

    fp32_bytes = model.SerializeToString()
    size_before_mb = len(fp32_bytes) / 1e6 if converted else args.model.stat().st_size / 1e6

    reader: CalibrationDataReader | None = None
    if args.dynamic:
        if args.calibration_images:
            print("ignoring --calibration-images: dynamic quantization does not need calibration data")
        print("dynamic quantization selected")
    else:
        images = resolve_calibration_images(args.calibration_images)
        if not images:
            raise SystemExit(
                "static quantization requires --calibration-images with at least one image (directory or "
                + "comma-separated paths); pass --dynamic to skip calibration"
            )
        print(f"static quantization selected; {len(images)} calibration image(s)")
        reader = ImageCalibrationReader(images, model.graph.input[0].name, args.channels, args.spatial, args.batch)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    quant_type = quantize_model(model, args.output, reader, dynamic=args.dynamic)
    print(f"wrote {args.output} ({quant_type})")

    size_after_mb = args.output.stat().st_size / 1e6
    ok, rows = validate_output(args.output, fp32_bytes, args.channels, args.spatial, args.batch, args.tolerance)
    print()
    for label, value in rows:
        print(f"  [{label}] {value}")

    if not ok:
        failing = [label for label, value in rows if value.startswith("FAIL")]
        print()
        print("FAILED VALIDATION GATES: " + ", ".join(failing))
        print(f"the int8 artifact {args.output} must not be deployed")
        raise SystemExit(1)

    print()
    print("ALL VALIDATION GATES PASSED")
    print()
    summary_rows: list[tuple[str, str]] = [
        ("model", str(args.model)),
        ("output", str(args.output)),
        ("quant type", quant_type),
    ]
    if converted:
        summary_rows.insert(1, ("fp16 -> fp32", f"converted {converted} tensor(s)"))
    summary_rows.extend(
        [
            ("input size (fp32)", f"{size_before_mb:.2f} MB"),
            ("output size (int8)", f"{size_after_mb:.2f} MB"),
            ("size reduction", f"{(1 - size_after_mb / size_before_mb) * 100:.1f}%"),
        ]
    )
    summary_rows.extend(rows)
    summary_rows.append(("overall", "PASS"))
    _print_table(summary_rows)


if __name__ == "__main__":
    main()
