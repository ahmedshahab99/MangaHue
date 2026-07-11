from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CONTRACT_ID = "mangahue.colorizer.v1"


@dataclass(frozen=True)
class ModelContract:
    contract_id: str = CONTRACT_ID
    input_name: str = "image_input"
    output_name: str = "color_output"
    input_channels: tuple[str, ...] = ("lineart", "hint_r", "hint_g", "hint_b", "hint_mask")
    output_channels: tuple[str, ...] = ("srgb_r", "srgb_g", "srgb_b")
    input_range: str = "0_1"
    output_range: str = "0_1"
    color_space: str = "srgb"
    lineart_inverted: bool = False
    dynamic_spatial_axes: bool = True
    spatial_multiple: int = 8

    def validate(self) -> None:
        if self.contract_id != CONTRACT_ID:
            raise ValueError(f"unsupported contract_id: {self.contract_id}")
        if self.input_channels != ("lineart", "hint_r", "hint_g", "hint_b", "hint_mask"):
            raise ValueError("input channel order is incompatible with the MangaHue v1 runtime")
        if self.output_channels != ("srgb_r", "srgb_g", "srgb_b"):
            raise ValueError("output channels must be ordered sRGB")
        if self.input_range != "0_1" or self.output_range != "0_1":
            raise ValueError("MangaHue v1 supports only normalized 0..1 tensors")
        if self.spatial_multiple <= 0:
            raise ValueError("spatial_multiple must be positive")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["input_channels"] = list(self.input_channels)
        value["output_channels"] = list(self.output_channels)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelContract":
        contract = cls(
            contract_id=value["contract_id"],
            input_name=value.get("input_name", "image_input"),
            output_name=value.get("output_name", "color_output"),
            input_channels=tuple(value["input_channels"]),
            output_channels=tuple(value["output_channels"]),
            input_range=value["input_range"],
            output_range=value["output_range"],
            color_space=value.get("color_space", "srgb"),
            lineart_inverted=bool(value.get("lineart_inverted", False)),
            dynamic_spatial_axes=bool(value.get("dynamic_spatial_axes", True)),
            spatial_multiple=int(value.get("spatial_multiple", 8)),
        )
        contract.validate()
        return contract


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_contract(path: Path) -> ModelContract:
    value = json.loads(path.read_text(encoding="utf-8"))
    return ModelContract.from_dict(value.get("contract", value))
