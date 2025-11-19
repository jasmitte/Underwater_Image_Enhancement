"""High-level orchestration of the multi-stage dehazing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .metrics import MetricResult, compute_metrics
from .postprocess import PostprocessConfig, postprocess_image
from .preprocess import PreprocessConfig, preprocess_image
from .utils import ensure_dir, load_image, save_image
from .zid import ZIDConfig, ZIDModel


@dataclass
class EnhancerConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    zid: ZIDConfig = field(default_factory=ZIDConfig)


class MultistageUnderwaterEnhancer:
    def __init__(self, config: EnhancerConfig | None = None):
        self.config = config or EnhancerConfig()
        self.zid = ZIDModel(self.config.zid)

    def enhance_array(self, image: np.ndarray) -> np.ndarray:
        preprocessed = preprocess_image(image, self.config.preprocess)
        dehazed, _ = self.zid.dehaze(preprocessed)
        postprocessed = postprocess_image(dehazed, self.config.postprocess)
        return np.clip(postprocessed, 0.0, 1.0)

    def enhance_file(self, input_path: Path, output_path: Path, reference_path: Optional[Path] = None) -> Dict[str, float]:
        image = load_image(input_path)
        enhanced = self.enhance_array(image)
        ensure_dir(output_path.parent)
        save_image(enhanced, output_path)
        metrics: Dict[str, float] = {}
        if reference_path:
            reference = load_image(reference_path)
            metric_values: MetricResult = compute_metrics(reference, enhanced)
            metrics = metric_values.as_dict()
        return metrics

    def batch_enhance(
        self,
        inputs: Iterable[Path],
        output_dir: Path,
        references: Optional[Iterable[Optional[Path]]] = None,
    ) -> List[Dict[str, float]]:
        ensure_dir(output_dir)
        input_list = list(inputs)
        ref_list = list(references) if references is not None else [None] * len(input_list)
        metrics_collection: List[Dict[str, float]] = []
        for idx, in_path in enumerate(input_list):
            ref = ref_list[idx] if idx < len(ref_list) else None
            out_path = output_dir / f"{in_path.stem}_enhanced.png"
            metrics = self.enhance_file(in_path, out_path, ref)
            metrics_collection.append(metrics)
        return metrics_collection
