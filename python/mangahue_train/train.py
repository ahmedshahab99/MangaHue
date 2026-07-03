from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument(
        "--synthetic-hints-prob",
        type=float,
        default=0.5,
        help="Probability of synthetic hints per sample when enabled",
    )
    return parser.parse_args()


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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, _train_ds = _build_loaders(args)

    from .models import build_model

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    l1 = nn.L1Loss()

    best_val: float | None = None
    best_path = args.best_output or args.output.with_name(args.output.stem + ".best.pt")

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = l1(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        train_loss = total / max(1, len(train_loader))
        log_line = f"epoch {epoch + 1}: train_loss={train_loss:.5f}"

        if val_loader is not None:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx = vx.to(device)
                    vy = vy.to(device)
                    vpred = model(vx)
                    val_total += float(l1(vpred, vy).detach().cpu())
            val_loss = val_total / max(1, len(val_loader))
            log_line += f" val_loss={val_loss:.5f}"
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "val_loss": val_loss, "epoch": epoch + 1}, best_path)
                log_line += f" (saved best -> {best_path})"
        print(log_line)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.cpu().state_dict()}, args.output)
    print(f"saved final checkpoint -> {args.output}")
    if best_val is not None:
        print(f"best val_loss={best_val:.5f} stored at {best_path}")


if __name__ == "__main__":
    main()
