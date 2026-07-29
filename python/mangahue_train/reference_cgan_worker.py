from __future__ import annotations

# Architecture adapted from linshys/Example_Based_Manga_Colorization---cGAN
# under Apache-2.0. The inference-only implementation omits VGG's unused
# classifier and caches the reference embedding once per worker.

import argparse
import json
import math
import os
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision import models


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.bottle_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.bottle_conv(value)
        return (self.double_conv(value) + value) / math.sqrt(2.0)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.double_conv(value)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 4, 2, 1),
            nn.LeakyReLU(0.1, True),
            ResBlock(in_channels, out_channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.main(value)


class StyleDynamicFilter(nn.Module):
    def __init__(self, color_dim: int, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.scale = 1.0 / math.sqrt(channels * kernel_size**2)
        self.modulation = nn.Conv2d(color_dim, channels, 1)
        self.weight = nn.Parameter(
            torch.randn(1, channels, channels, kernel_size, kernel_size)
        )

    def forward(
        self, features: torch.Tensor, color_style: torch.Tensor
    ) -> torch.Tensor:
        batch, channels, height, width = features.shape
        style = self.modulation(color_style).view(batch, 1, channels, 1, 1)
        weight = self.scale * self.weight * style
        demodulation = torch.rsqrt(weight.pow(2).sum((2, 3, 4)) + 1e-8)
        weight = weight * demodulation.view(batch, channels, 1, 1, 1)
        weight = weight.view(
            batch * channels, channels, self.kernel_size, self.kernel_size
        )
        features = features.view(1, batch * channels, height, width)
        features = functional.conv2d(
            features, weight, padding=self.padding, groups=batch
        )
        return features.view(batch, channels, height, width)


class UpBlock(nn.Module):
    def __init__(
        self,
        color_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(in_channels // 2 + in_channels // 8, out_channels, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.conv_s = nn.Conv2d(in_channels // 2, out_channels, 1)
        self.SDFT = StyleDynamicFilter(color_dim, out_channels, kernel_size)

    def forward(
        self,
        value: torch.Tensor,
        skip: torch.Tensor,
        color_style: torch.Tensor,
    ) -> torch.Tensor:
        value = self.up(value)
        shortcut = self.conv_s(value)
        value = torch.cat((value, skip[:, ::4, :, :]), dim=1)
        value = self.conv_cat(value)
        return self.SDFT(value, color_style) + shortcut


class ColorUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = UpBlock(512, 1024, 256, 3)
        self.up2 = UpBlock(512, 512, 128, 3)
        self.up3 = UpBlock(512, 256, 64, 5)
        self.up4 = UpBlock(512, 128, 64, 5)
        self.outc = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.Tanh(),
        )

    def forward(
        self, lightness: torch.Tensor, color_style: torch.Tensor
    ) -> torch.Tensor:
        level1 = self.inc(lightness)
        level2 = self.down1(level1)
        level3 = self.down2(level2)
        level4 = self.down3(level3)
        level5 = self.down4(level4)
        value = self.up1(level5, level4, color_style)
        value = self.up2(value, level3, color_style)
        value = self.up3(value, level2, color_style)
        value = self.up4(value, level1, color_style)
        return self.outc(value)


class ReferenceEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vgg_features = models.vgg19(weights=None).features
        self.feature2vector = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 4, 2, 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        standard_deviation = image.new_tensor((0.229, 0.224, 0.225)).view(
            1, 3, 1, 1
        )
        value = (image - mean) / standard_deviation
        for index, layer in enumerate(self.vgg_features):
            value = layer(value)
            if index == 31:
                break
        return self.feature2vector(value)


class ReferenceColorizer:
    def __init__(
        self,
        checkpoint_path: Path,
        reference_path: Path,
        size: int,
        threads: int,
    ) -> None:
        torch.set_num_threads(max(1, threads))
        torch.set_num_interop_threads(1)
        self.size = size
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )

        encoder_state = checkpoint["colorEncoder"]
        self.encoder = ReferenceEncoder()
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
        self.encoder.vgg_features.load_state_dict(vgg_state)
        self.encoder.feature2vector.load_state_dict(vector_state)

        self.colorizer = ColorUNet()
        self.colorizer.load_state_dict(checkpoint["colorUNet"])
        del checkpoint, encoder_state, vgg_state, vector_state
        self.encoder.eval()
        self.colorizer.eval()

        reference = Image.open(reference_path).convert("RGB")
        reference_array = np.asarray(reference, dtype=np.float32) / 255.0
        reference_tensor = torch.from_numpy(reference_array.transpose(2, 0, 1))
        reference_tensor = reference_tensor.unsqueeze(0)
        with torch.inference_mode():
            resized = functional.interpolate(
                reference_tensor,
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            )
            self.style = self.encoder(resized)
        del self.encoder

    def colorize(self, lineart_path: Path, output_path: Path) -> None:
        source = Image.open(lineart_path).convert("RGB")
        source_array = np.asarray(source, dtype=np.float32) / 255.0
        lab = color.rgb2lab(source_array).astype(np.float32)
        lightness = torch.from_numpy(
            ((lab[:, :, :1] - 50.0) / 50.0).transpose(2, 0, 1)
        ).unsqueeze(0)
        with torch.inference_mode():
            resized = functional.interpolate(
                lightness,
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False,
            )
            ab_channels = self.colorizer(resized, self.style)
            ab_channels = functional.interpolate(
                ab_channels * 110.0,
                size=(source.height, source.width),
                mode="bilinear",
                align_corners=False,
            )
        output_lab = np.concatenate(
            (
                lab[:, :, :1],
                ab_channels[0].cpu().numpy().transpose(1, 2, 0),
            ),
            axis=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            output = (
                np.clip(color.lab2rgb(output_lab), 0.0, 1.0) * 255.0
            ).astype(np.uint8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(output).save(output_path)


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    colorizer = ReferenceColorizer(
        args.model, args.reference, args.size, args.threads
    )
    _emit({"status": "ready", "backend": "reference-cgan"})

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        request: dict[str, object] = {}
        try:
            request = json.loads(raw_line)
            colorizer.colorize(
                Path(str(request["lineart"])),
                Path(str(request["output"])),
            )
            _emit({"ok": True, "id": request.get("id")})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            _emit(
                {
                    "ok": False,
                    "id": request.get("id"),
                    "error": str(error),
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
