from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import MangaHueDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MangaHue Hint U-Net")
    parser.add_argument("--manifest", type=Path, required=True, help="Training manifest JSON")
    parser.add_argument("--val-manifest", type=Path, help="Optional validation manifest JSON")
    parser.add_argument("--output", type=Path, required=True, help="Final checkpoint path")
    parser.add_argument("--best-output", type=Path, help="Best-by-val-loss checkpoint path")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, help="If --val-manifest not given, hold out this fraction of --manifest for validation")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--synthetic-hints", action="store_true", help="Generate sparse hint dots from color targets when no hint JSON is present")
    parser.add_argument("--resume", type=Path, help="Resume model, optimizer, scaler, EMA, and epoch state")
    parser.add_argument("--accumulate", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--edge-weight", type=float, default=0.2)
    parser.add_argument("--luminance-weight", type=float, default=0.15)
    parser.add_argument(
        "--synthetic-hints-prob",
        type=float,
        default=0.5,
        help="Probability of synthetic hints per sample when enabled",
    )
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _gradient(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return image[..., :, 1:] - image[..., :, :-1], image[..., 1:, :] - image[..., :-1, :]


def _training_loss(pred: torch.Tensor, target: torch.Tensor, edge_weight: float, luminance_weight: float) -> torch.Tensor:
    color = torch.nn.functional.l1_loss(pred, target)
    pred_dx, pred_dy = _gradient(pred)
    target_dx, target_dy = _gradient(target)
    edge = torch.nn.functional.l1_loss(pred_dx, target_dx) + torch.nn.functional.l1_loss(pred_dy, target_dy)
    weights = pred.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    luminance = torch.nn.functional.l1_loss((pred * weights).sum(1), (target * weights).sum(1))
    return color + edge_weight * edge + luminance_weight * luminance


@torch.no_grad()
def _update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for target, source in zip(ema.parameters(), model.parameters(), strict=True):
        target.mul_(decay).add_(source.detach(), alpha=1.0 - decay)
    for target, source in zip(ema.buffers(), model.buffers(), strict=True):
        target.copy_(source)


def _build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None, MangaHueDataset]:
    if args.val_manifest:
        train_ds = MangaHueDataset(
            args.manifest,
            size=args.size,
            synthetic_hints={"min_dots": 0, "max_dots": 8, "radius": 0.025, "prob": args.synthetic_hints_prob}
            if args.synthetic_hints
            else False,
            seed=args.seed,
        )
        val_ds: MangaHueDataset | None = MangaHueDataset(
            args.val_manifest,
            size=args.size,
            synthetic_hints=False,
            seed=args.seed + 1,
        )
    elif args.val_ratio and args.val_ratio > 0:
        samples = MangaHueDataset(args.manifest, size=args.size).samples
        cut = max(1, int(round(len(samples) * args.val_ratio)))
        val_samples = samples[:cut]
        train_samples = samples[cut:]
        train_ds = MangaHueDataset(
            train_samples,
            size=args.size,
            synthetic_hints={"min_dots": 0, "max_dots": 8, "radius": 0.025, "prob": args.synthetic_hints_prob}
            if args.synthetic_hints
            else False,
            seed=args.seed,
        )
        val_ds = MangaHueDataset(
            val_samples,
            size=args.size,
            synthetic_hints=False,
            seed=args.seed + 1,
        )
    else:
        train_ds = MangaHueDataset(
            args.manifest,
            size=args.size,
            synthetic_hints={"min_dots": 0, "max_dots": 8, "radius": 0.025, "prob": args.synthetic_hints_prob}
            if args.synthetic_hints
            else False,
            seed=args.seed,
        )
        val_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        if val_ds is not None
        else None
    )
    return train_loader, val_loader, train_ds


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, _train_ds = _build_loaders(args)

    from .models import build_model

    model = build_model().to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp_enabled = bool(args.amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    l1 = nn.L1Loss()

    best_val: float | None = None
    start_epoch = 0
    best_path = args.best_output or args.output.with_name(args.output.stem + ".best.pt")
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        ema_model.load_state_dict(state.get("ema_model", state["model"]))
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", 0))
        best_val = state.get("best_val")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"), start=1):
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp_enabled):
                pred = model(x)
                loss = _training_loss(pred, y, args.edge_weight, args.luminance_weight)
                scaled_loss = loss / max(1, args.accumulate)
            scaler.scale(scaled_loss).backward()
            if step % max(1, args.accumulate) == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                _update_ema(ema_model, model, args.ema_decay)
            total += float(loss.detach().cpu())
        train_loss = total / max(1, len(train_loader))
        log_line = f"epoch {epoch + 1}: train_loss={train_loss:.5f}"

        if val_loader is not None:
            ema_model.eval()
            val_total = 0.0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx = vx.to(device)
                    vy = vy.to(device)
                    vpred = ema_model(vx)
                    val_total += float(l1(vpred, vy).detach().cpu())
            val_loss = val_total / max(1, len(val_loader))
            log_line += f" val_loss={val_loss:.5f}"
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": ema_model.state_dict(), "val_loss": val_loss, "epoch": epoch + 1}, best_path)
                log_line += f" (saved best -> {best_path})"
        print(log_line)
        latest_path = args.output.with_name(args.output.stem + ".latest.pt")
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(), "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch + 1, "best_val": best_val, "seed": args.seed,
        }, latest_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": ema_model.cpu().state_dict(), "best_val": best_val, "seed": args.seed}, args.output)
    print(f"saved final checkpoint -> {args.output}")
    if best_val is not None:
        print(f"best val_loss={best_val:.5f} stored at {best_path}")


if __name__ == "__main__":
    main()
