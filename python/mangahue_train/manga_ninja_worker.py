from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SIZE: int = 512


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent MangaNinja colorizer worker")
    parser.add_argument("--model-root", type=Path, required=True,
                        help="Root of the MangaNinja-Pretrained tree")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--denoise-steps", type=int, default=50)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def _validate_model_root(root: Path) -> None:
    root = root.resolve()
    required_dirs = (
        root / "checkpoints" / "StableDiffusion",
        root / "checkpoints" / "models",
        root / "checkpoints" / "MangaNinjia",
    )
    missing = [str(d) for d in required_dirs if not d.is_dir()]
    if missing:
        raise FileNotFoundError("missing checkpoint directories: " + ", ".join(missing))

    required_files = (
        root / "src" / "models" / "unet_2d_condition.py",
        root / "src" / "models" / "refunet_2d_condition.py",
        root / "src" / "models" / "mutual_self_attention_multi_scale.py",
        root / "src" / "point_network.py",
        root / "src" / "annotator" / "lineart.py",
        root / "inference" / "manganinjia_pipeline.py",
    )
    missing_files = [str(f) for f in required_files if not f.is_file()]
    if missing_files:
        raise FileNotFoundError("missing source files: " + ", ".join(missing_files))


def _set_env(threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ["MALLOC_ARENA_MAX"] = "2"
    torch.set_num_threads(max(1, threads))
    torch.set_num_interop_threads(1)


def _composite_reference(hint_map_path: str, hint_mask_path: str) -> np.ndarray:
    hint_map = np.array(Image.open(hint_map_path).convert("RGB"))
    hint_mask = np.array(Image.open(hint_mask_path).convert("L"))
    return np.where(hint_mask[:, :, None] > 0, hint_map, 255).astype(np.uint8)


def _ensure_paths(root: Path) -> tuple[str, str]:
    src_path = str(root / "src")
    root_path = str(root)
    for p in (src_path, root_path):
        if p not in sys.path:
            sys.path.insert(0, p)
    return src_path, root_path


def _load_pipeline(model_root: Path, device: str, denoise_steps: int):  # noqa: C901
    model_root = model_root.resolve()
    _ensure_paths(model_root)

    import diffusers  # noqa: F811
    from diffusers import AutoencoderKL, ControlNetModel, DDIMScheduler
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    from src.models.unet_2d_condition import UNet2DConditionModel
    from src.models.refunet_2d_condition import RefUNet2DConditionModel
    from src.point_network import PointNet
    from inference.manganinjia_pipeline import MangaNinjaPipeline

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    sd_root = model_root / "checkpoints" / "StableDiffusion"
    models_root = model_root / "checkpoints" / "models"

    vae = AutoencoderKL.from_pretrained(
        sd_root, subfolder="vae", torch_dtype=torch_dtype,
    ).to(device).eval()

    clip_vision_1 = CLIPVisionModelWithProjection.from_pretrained(
        sd_root, subfolder="image_encoder", torch_dtype=torch_dtype,
    ).to(device).eval()

    clip_vision_2 = CLIPVisionModelWithProjection.from_pretrained(
        sd_root, subfolder="image_encoder", torch_dtype=torch_dtype,
    ).to(device).eval()

    clip_processor = CLIPImageProcessor.from_pretrained(
        sd_root, subfolder="image_encoder",
    )

    controlnet = ControlNetModel.from_pretrained(
        models_root, subfolder="controlnet", torch_dtype=torch_dtype,
    ).to(device).eval()

    unet = UNet2DConditionModel.from_pretrained(
        models_root, subfolder="unet", torch_dtype=torch_dtype,
    ).to(device).eval()

    ref_unet = RefUNet2DConditionModel.from_pretrained(
        models_root, subfolder="ref_unet", torch_dtype=torch_dtype,
    ).to(device).eval()

    point_net_state = torch.load(
        models_root / "point_net" / "pointnet.pth", map_location=device, weights_only=True,
    )
    point_net = PointNet()
    point_net.load_state_dict(point_net_state)
    point_net.to(device).eval()

    for attn_name in ("attn1", "attn2", "attn3"):
        unet.set_attn_processor(attn_name, None)
        ref_unet.set_attn_processor(attn_name, None)

    scheduler = DDIMScheduler.from_pretrained(sd_root, subfolder="scheduler")

    pipeline = MangaNinjaPipeline(
        vae=vae,
        unet=unet,
        ref_unet=ref_unet,
        controlnet=controlnet,
        image_encoder=clip_vision_1,
        image_encoder_2=clip_vision_2,
        feature_extractor=clip_processor,
        point_net=point_net,
        scheduler=scheduler,
        device=device,
    )
    pipeline.set_progress_bar_config(disable=True)

    return pipeline


def _cleanup_pipeline(pipeline) -> None:
    for attr in ("current_image", "current_hint", "_attention_control", "_ref_control"):
        if hasattr(pipeline, attr):
            setattr(pipeline, attr, None)


def _resize_to_native(image: np.ndarray, interp: int = Image.Resampling.LANCZOS) -> Image.Image:
    return Image.fromarray(image).resize((SIZE, SIZE), interp)


def _run_request(pipeline, request: dict[str, object], denoise_steps: int, device: str) -> dict[str, object]:
    lineart_path = str(request["lineart"])
    hint_map_path = str(request["hint_map"])
    hint_mask_path = str(request["hint_mask"])
    output_path = Path(str(request["output"]))

    lineart = np.array(Image.open(lineart_path).convert("L"))
    reference = _composite_reference(hint_map_path, hint_mask_path)

    with Image.open(lineart_path) as src:
        orig_w, orig_h = src.size

    ref_pil = _resize_to_native(reference, Image.Resampling.LANCZOS)
    line_pil = _resize_to_native(lineart, Image.Resampling.NEAREST).convert("RGB")

    with torch.inference_mode():
        result = pipeline(
            is_lineart=True,
            ref_image=ref_pil,
            target_image=line_pil,
            num_inference_steps=denoise_steps,
        )

    if hasattr(result, "images") and isinstance(result.images, (list, tuple)):
        output_pil = result.images[0]
    elif hasattr(result, "images"):
        output_pil = result.images
    elif isinstance(result, Image.Image):
        output_pil = result
    elif isinstance(result, np.ndarray):
        arr = np.clip(result, 0, 255).astype(np.uint8)
        output_pil = Image.fromarray(arr)
    elif isinstance(result, torch.Tensor):
        arr = result.detach().cpu().float().numpy()
        arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        output_pil = Image.fromarray(arr)
    else:
        raise TypeError(f"unexpected pipeline output type: {type(result)}")

    if output_pil.size != (orig_w, orig_h):
        output_pil = output_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_pil.save(output_path, format="PNG", optimize=False)

    _cleanup_pipeline(pipeline)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"ok": True, "id": request.get("id"), "width": orig_w, "height": orig_h}


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    _set_env(args.threads)
    _validate_model_root(args.model_root)
    pipeline = _load_pipeline(args.model_root, args.device, args.denoise_steps)

    print(json.dumps({"status": "ready", "backend": "manga-ninja-v1", "size": SIZE}), flush=True)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = _run_request(pipeline, request, args.denoise_steps, args.device)
        except Exception as error:
            response = {
                "ok": False,
                "error": str(error),
                "traceback": traceback.format_exc(limit=8),
            }
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
