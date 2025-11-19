"""Pre-processing utilities described in the reference paper."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    use_clahe: bool = True
    clip_limit: float = 2.0
    tile_grid: int = 8
    gamma: float = 1.05
    gray_world: bool = True
    red_boost: float = 0.05


def _apply_clahe(image: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    lab = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return result


def _gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(np.clip(image, 1e-4, 1.0), gamma)


def _gray_world_balance(image: np.ndarray) -> np.ndarray:
    mean = image.mean(axis=(0, 1), keepdims=True)
    gray = mean.mean()
    scale = gray / (mean + 1e-6)
    balanced = np.clip(image * scale, 0.0, 1.0)
    return balanced


def _restore_red_channel(image: np.ndarray, boost: float) -> np.ndarray:
    restored = image.copy()
    restored[:, :, 0] = np.clip(restored[:, :, 0] + boost, 0.0, 1.0)
    return restored


def preprocess_image(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    output = image.copy()
    if config.gray_world:
        output = _gray_world_balance(output)
    if config.red_boost > 0:
        output = _restore_red_channel(output, config.red_boost)
    if config.use_clahe:
        output = _apply_clahe(output, config.clip_limit, config.tile_grid)
    if config.gamma != 1.0:
        output = _gamma_correction(output, config.gamma)
    return np.clip(output, 0.0, 1.0)