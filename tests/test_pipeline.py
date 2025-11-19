"""Smoke tests for the multistage dehazing pipeline."""

from __future__ import annotations

import numpy as np

from src.pipeline import EnhancerConfig, MultistageUnderwaterEnhancer
from src.zid import ZIDConfig


def _make_synthetic_scene(size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, size, dtype=np.float32)
    clean = np.dstack(np.meshgrid(x, x))
    clean = np.concatenate([clean, clean[:, :, :1]], axis=2)
    transmission = np.clip(np.outer(x, x).reshape(size, size), 0.4, 1.0)
    atmosphere = np.array([0.8, 0.9, 1.0], dtype=np.float32)
    hazy = clean * transmission[..., None] + atmosphere * (1 - transmission[..., None])
    return clean, np.clip(hazy, 0.0, 1.0)


def test_pipeline_enhances_image():
    reference, hazy = _make_synthetic_scene()
    config = EnhancerConfig(zid=ZIDConfig(max_iterations=2, lr=5e-3))
    enhancer = MultistageUnderwaterEnhancer(config)
    enhanced = enhancer.enhance_array(hazy)
    assert enhanced.shape == reference.shape
    assert np.isfinite(enhanced).all()
    assert (0.0 <= enhanced).all() and (enhanced <= 1.0).all()
