"""Pre-processing utilities described in the reference paper.

According to "An Enhanced Multi-Stage Approach for Dehazing Underwater Images"
(Murugan et al., 2024), Section III.A, the preprocessing stage should ONLY apply
CLAHE (Contrast Limited Adaptive Histogram Equalization) for initial contrast
enhancement before the ZID dehazing step.

The paper specifically states to use "histogram equalization or CLAHE" - meaning
ONE contrast enhancement method, not multiple preprocessing steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing stage.

    As per the An_Enhanced paper Section III.A (lines 948-978), preprocessing
    should perform either histogram equalization OR CLAHE for initial contrast
    enhancement before dehazing. CLAHE is preferred to avoid over-enhancement.
    """
    method: str = "clahe"  # "clahe" or "histogram_equalization"
    clip_limit: float = 2.0  # For CLAHE only
    tile_grid: int = 8  # For CLAHE only


def _apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply standard Histogram Equalization.

    As mentioned in Section III.A (lines 971-974): "This method improves the
    contrasted image which effectively spreads out the most frequent values of
    intensity. However, HE can sometimes lead to over-enhancement, making the
    details in bright or dark areas less visible."

    Applied in LAB color space to the L channel to preserve color.

    Args:
        image: Input image in RGB format, normalized to [0, 1]

    Returns:
        Histogram-equalized image in RGB format, normalized to [0, 1]
    """
    lab = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return result


def _apply_clahe(image: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization.

    As mentioned in Section III.A (lines 975-978): "Adaptive Histogram
    Equalization (AHE) or its variant, Contrast Limited AHE (CLAHE), are often
    preferred as they perform histogram equalization locally, reducing the risk
    of over-enhancement."

    This is performed in LAB color space to preserve color information
    while enhancing contrast through the L (lightness) channel.

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
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return result


def preprocess_image(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Preprocess underwater image before ZID dehazing.

    Following Section III.A of the An_Enhanced paper (lines 948-978), this
    function applies either histogram equalization OR CLAHE for initial contrast
    enhancement. The paper states (lines 227-229):

    "pre-processing techniques like contrast enhancement and histogram
    equalization followed by zero-shot dehazing"

    All other color corrections (white balance, gamma correction, saturation
    adjustment) are performed in the post-processing stage AFTER dehazing,
    as specified in Section III.B.

    Args:
        image: Input underwater image in RGB format, normalized to [0, 1]
        config: Preprocessing configuration

    Returns:
        Preprocessed image ready for ZID dehazing, normalized to [0, 1]
    """
    if config.method == "clahe":
        return _apply_clahe(image, config.clip_limit, config.tile_grid)
    elif config.method == "histogram_equalization":
        return _apply_histogram_equalization(image)
    else:
        # No preprocessing
        return image.copy()