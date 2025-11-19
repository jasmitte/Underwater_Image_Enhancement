"""Shared helpers for file IO and numeric conversions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image


@dataclass
class ImageBatch:
    """Container for batching enhancement results."""

    paths: List[Path]
    images: List[np.ndarray]

    def __iter__(self) -> Iterable[Tuple[Path, np.ndarray]]:
        return zip(self.paths, self.images)


def load_image(path: Path) -> np.ndarray:
    """Load an RGB image normalized to [0, 1]."""

    with Image.open(path) as img:
        rgb = img.convert("RGB")
    return np.asarray(rgb, dtype=np.float32) / 255.0


def save_image(image: np.ndarray, path: Path) -> None:
    """Persist normalized RGB data to disk."""

    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def from_uint8(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0
