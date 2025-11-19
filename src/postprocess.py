"""Post-processing logic for color correction and contrast polishing."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PostprocessConfig:
    use_white_balance: bool = True
    saturation_boost: float = 1.15
    apply_clahe: bool = True
    clip_limit: float = 2.0
    tile_grid: int = 8
    auto_contrast: bool = True


def _white_balance(image: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split((image * 255.0).astype(np.uint8))
    result = cv2.merge((cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)))
    return result.astype(np.float32) / 255.0


def _boost_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    merged = cv2.merge((h, s, v))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0


def _apply_clahe(image: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    lab = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def _auto_contrast(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    stretched = np.clip((image - low) / (high - low + 1e-6), 0.0, 1.0)
    return stretched


def postprocess_image(image: np.ndarray, config: PostprocessConfig) -> np.ndarray:
    output = image.copy()
    if config.use_white_balance:
        output = _white_balance(output)
    if config.saturation_boost != 1.0:
        output = _boost_saturation(output, config.saturation_boost)
    if config.apply_clahe:
        output = _apply_clahe(output, config.clip_limit, config.tile_grid)
    if config.auto_contrast:
        output = _auto_contrast(output)
    return np.clip(output, 0.0, 1.0)