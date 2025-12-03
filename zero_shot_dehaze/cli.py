"""Command line interface for running the enhancer on datasets or single images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from .pipeline import EnhancerConfig, MultistageUnderwaterEnhancer
from .zid import ZIDConfig

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _collect_images(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted([p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension for {path}")
    return [path]


def _collect_references(paths: Iterable[Path], reference_root: Optional[Path]) -> List[Optional[Path]]:
    path_list = list(paths)
    if reference_root is None:
        return [None] * len(path_list)
    ref_list: List[Optional[Path]] = []
    for img in path_list:
        candidate = reference_root / img.name if reference_root.is_dir() else reference_root
        ref_list.append(candidate if candidate.exists() else None)
    return ref_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced multi-stage underwater dehazing")
    parser.add_argument("--input", type=Path, required=True, help="Path to an image or directory")
    parser.add_argument("--output", type=Path, required=True, help="Directory for enhanced results")
    parser.add_argument("--reference", type=Path, default=None, help="Optional ground-truth directory or file")
    parser.add_argument("--device", type=str, default=None, help="Force a device id (cpu, cuda, cuda:1, ...)")
    parser.add_argument("--max-iterations", type=int, default=600, help="Optimization steps for the ZID stage")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE in pre and post stages")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = _collect_images(args.input)
    references = _collect_references(inputs, args.reference)

    zid_config = ZIDConfig(max_iterations=args.max_iterations, device=args.device)
    enhancer = MultistageUnderwaterEnhancer(EnhancerConfig(zid=zid_config))

    if args.no_clahe:
        enhancer.config.preprocess.use_clahe = False
        enhancer.config.postprocess.apply_clahe = False

    metrics = enhancer.batch_enhance(inputs, args.output, references)
    if any(metrics):
        for path, metric in zip(inputs, metrics):
            if metric:
                values = ", ".join(f"{k}: {v:.3f}" for k, v in metric.items())
                print(f"{path.name}: {values}")


if __name__ == "__main__":
    main()