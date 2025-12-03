"""Post-processing logic for color correction and contrast polishing.

According to "An Enhanced Multi-Stage Approach for Dehazing Underwater Images"
(Murugan et al., 2024), Section III.B, the post-processing stage should include:
1. White balance correction (gray world algorithm)
2. Adaptive saturation adjustment (based on image characteristics)
3. Gamma correction for brightness adjustment
4. CLAHE for final contrast enhancement
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PostprocessConfig:
    """Configuration for post-processing stage.

    As per the An_Enhanced paper Section III.B, post-processing includes:
    - White balance correction
    - Adaptive saturation enhancement
    - Gamma correction
    - CLAHE contrast enhancement
    """
    use_white_balance: bool = True
    white_balance_method: str = "gray_world"  # gray_world or simple_white_patch
    adaptive_saturation: bool = True
    base_saturation_factor: float = 1.2
    use_gamma_correction: bool = True
    gamma: float = 1.2
    apply_clahe: bool = True
    clip_limit: float = 2.0
    tile_grid: int = 8


def _white_balance_gray_world(image: np.ndarray) -> np.ndarray:
    """Apply Gray World white balance algorithm.

    This method assumes the average color in the scene is gray. It adjusts
    each color channel by its mean to achieve white balance, which is the
    standard approach mentioned in the paper (Section III.B, lines 1280-1300).

    Args:
        image: Input image in RGB format, normalized to [0, 1]

    Returns:
        White-balanced image in RGB format, normalized to [0, 1]
    """
    # Convert to float for calculations
    result = (image * 255.0).astype(np.float32)

    # Calculate mean for each channel
    avg_r = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_b = np.mean(result[:, :, 2])

    # Calculate gray average across all channels
    gray_avg = (avg_r + avg_g + avg_b) / 3.0

    # Scale each channel
    if avg_r > 0:
        result[:, :, 0] = np.clip(result[:, :, 0] * (gray_avg / avg_r), 0, 255)
    if avg_g > 0:
        result[:, :, 1] = np.clip(result[:, :, 1] * (gray_avg / avg_g), 0, 255)
    if avg_b > 0:
        result[:, :, 2] = np.clip(result[:, :, 2] * (gray_avg / avg_b), 0, 255)

    return result / 255.0


def _white_balance_simple_white_patch(image: np.ndarray) -> np.ndarray:
    """Apply Simple White Patch white balance algorithm.

    This method identifies the brightest areas and adjusts colors assuming
    those areas should be white/neutral (Section III.B, lines 1289-1294).

    Args:
        image: Input image in RGB format, normalized to [0, 1]

    Returns:
        White-balanced image in RGB format, normalized to [0, 1]
    """
    result = (image * 255.0).astype(np.float32)

    # Find brightest areas (top 1% of pixels by luminance)
    luminance = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
    threshold = np.percentile(luminance, 99)
    bright_mask = luminance >= threshold

    if np.sum(bright_mask) > 0:
        # Calculate max values from bright areas
        max_r = np.max(result[:, :, 0][bright_mask])
        max_g = np.max(result[:, :, 1][bright_mask])
        max_b = np.max(result[:, :, 2][bright_mask])

        # Scale to make brightest areas white
        if max_r > 0:
            result[:, :, 0] = np.clip(result[:, :, 0] * (255.0 / max_r), 0, 255)
        if max_g > 0:
            result[:, :, 1] = np.clip(result[:, :, 1] * (255.0 / max_g), 0, 255)
        if max_b > 0:
            result[:, :, 2] = np.clip(result[:, :, 2] * (255.0 / max_b), 0, 255)

    return result / 255.0


def _adaptive_saturation_adjustment(image: np.ndarray, base_factor: float = 1.2) -> np.ndarray:
    """Apply adaptive saturation adjustment based on image characteristics.

    As mentioned in the paper (Section III.B, lines 1310-1313): "Care must be
    taken to avoid over-saturation". This function analyzes the current saturation
    levels and applies adjustment adaptively to avoid over-saturation.

    Args:
        image: Input image in RGB format, normalized to [0, 1]
        base_factor: Base saturation boost factor (default: 1.2)

    Returns:
        Saturation-adjusted image in RGB format, normalized to [0, 1]
    """
    hsv = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # Analyze current saturation levels
    mean_saturation = np.mean(s)
    std_saturation = np.std(s)

    # Adaptive factor: reduce boost if image is already saturated
    # If mean saturation > 100 (out of 255), reduce the boost
    if mean_saturation > 100:
        adaptive_factor = base_factor * (1.0 - (mean_saturation - 100) / 155.0 * 0.5)
        adaptive_factor = max(1.0, adaptive_factor)  # Don't go below 1.0
    else:
        # For low saturation images, apply full boost
        adaptive_factor = base_factor

    # Apply adaptive saturation boost with per-pixel limiting
    s_float = s.astype(np.float32) * adaptive_factor

    # Soft clipping to avoid harsh over-saturation (mentioned in paper)
    # Use a sigmoid-like curve near the upper limit
    s_float = np.where(s_float > 200,
                       200 + (s_float - 200) * 0.3,  # Soft limit above 200
                       s_float)

    s = np.clip(s_float, 0, 255).astype(np.uint8)

    merged = cv2.merge((h, s, v))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0


def _gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Apply gamma correction for brightness adjustment.

    As specified in the paper abstract and Section III.B (lines 227-236, 990-999):
    "white balancing and gamma correction" are key post-processing steps.
    Gamma correction adjusts brightness non-linearly.

    Args:
        image: Input image in RGB format, normalized to [0, 1]
        gamma: Gamma value (>1 brightens, <1 darkens). Default: 1.2

    Returns:
        Gamma-corrected image in RGB format, normalized to [0, 1]
    """
    # Apply gamma correction: out = in^(1/gamma)
    inv_gamma = 1.0 / gamma
    corrected = np.power(image, inv_gamma)
    return np.clip(corrected, 0.0, 1.0)


def _apply_clahe(image: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    As specified in Section III.B (lines 1325-1336), CLAHE is preferred over
    AHE because it "limits the contrast amplification to prevent the noise from
    becoming overly enhanced in relatively homogeneous areas."

    This is applied in LAB color space to enhance the L (lightness) channel
    while preserving color information.

    Args:
        image: Input image in RGB format, normalized to [0, 1]
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_grid: Size of grid for histogram equalization (default: 8x8)

    Returns:
        CLAHE-enhanced image in RGB format, normalized to [0, 1]
    """
    lab = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def postprocess_image(image: np.ndarray, config: PostprocessConfig) -> np.ndarray:
    """Apply post-processing color correction and contrast enhancement.

    Following Section III.B of the An_Enhanced paper, this function applies:
    1. White balance correction to neutralize color cast (lines 1280-1300)
    2. Adaptive saturation adjustment to restore color vibrancy (lines 1310-1313)
    3. Gamma correction for brightness adjustment (lines 227-236, 990-999)
    4. CLAHE for final contrast enhancement (lines 1325-1336)

    The order of operations follows the paper's recommended sequence for
    optimal underwater image enhancement.

    Args:
        image: Dehazed image from ZID model in RGB format, normalized to [0, 1]
        config: Post-processing configuration

    Returns:
        Enhanced image with color correction applied, normalized to [0, 1]
    """
    output = image.copy()

    # Step 1: White balance correction (Section III.B.1)
    if config.use_white_balance:
        if config.white_balance_method == "gray_world":
            output = _white_balance_gray_world(output)
        elif config.white_balance_method == "simple_white_patch":
            output = _white_balance_simple_white_patch(output)

    # Step 2: Adaptive saturation adjustment (Section III.B.2)
    if config.adaptive_saturation:
        output = _adaptive_saturation_adjustment(output, config.base_saturation_factor)

    # Step 3: Gamma correction (mentioned in abstract and Section III.B)
    if config.use_gamma_correction:
        output = _gamma_correction(output, config.gamma)

    # Step 4: CLAHE contrast enhancement (Section III.B.3)
    if config.apply_clahe:
        output = _apply_clahe(output, config.clip_limit, config.tile_grid)

    return np.clip(output, 0.0, 1.0)