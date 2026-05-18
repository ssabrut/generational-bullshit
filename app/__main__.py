"""Command-line entrypoint: ``python -m app <input.png> [--out-dir DIR]``.

Runs :class:`FloorPlanPreprocessor` on a single image and writes the
text-free colour crop and the binary crop to disk.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from app.config import PreprocessorConfig
from app.pipeline import FloorPlanPreprocessor


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="Preprocess a floor-plan image (text removal + binarisation + crop).",
    )
    parser.add_argument("input", type=Path, help="Path to the input image.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/PNG"),
        help="Output root. Writes to <out-dir>/no_text/ and <out-dir>/crop_and_pad/.",
    )
    parser.add_argument("--angle-step", type=int, default=90)
    parser.add_argument("--no-gpu", action="store_true", help="Disable CUDA in EasyOCR.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    image = cv2.imread(str(args.input))
    if image is None:
        print(f"error: could not read image: {args.input}", file=sys.stderr)
        return 1

    config = PreprocessorConfig(angle_step=args.angle_step, gpu=not args.no_gpu)
    pipeline = FloorPlanPreprocessor(config)
    result = pipeline.process(image)

    no_text_dir = args.out_dir / "no_text"
    crop_dir = args.out_dir / "crop_and_pad"
    no_text_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    stem = args.input.stem
    cv2.imwrite(str(no_text_dir / f"{stem}.png"), result.raw_no_text)
    cv2.imwrite(str(crop_dir / f"{stem}.png"), result.preprocessed)

    print(f"rotation_angle : {result.rotation_angle}°")
    print(f"crop_bbox      : {result.crop_bbox}")
    print(f"Saved raw_no_text  → {no_text_dir / f'{stem}.png'}")
    print(f"Saved preprocessed → {crop_dir / f'{stem}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
