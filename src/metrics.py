"""Evaluation metrics described in the paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class MetricResult:
    psnr: float
    ssim: float

    def as_dict(self) -> Dict[str, float]:
        return {"psnr": self.psnr, "ssim": self.ssim}


def compute_metrics(reference: np.ndarray, estimate: np.ndarray) -> MetricResult:
    psnr_value = peak_signal_noise_ratio(reference, estimate, data_range=1.0)
    ssim_value = structural_similarity(reference, estimate, channel_axis=-1, data_range=1.0)
    return MetricResult(psnr=psnr_value, ssim=ssim_value)
