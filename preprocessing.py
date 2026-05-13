from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import easyocr


@dataclass
class PipelineResult:
    """
    raw_no_text    : BGR image — text removed, NOT cropped (full original size).
    preprocessed   : binary image — Gaussian blur + Otsu, cropped to content bbox.
    rotation_angle : clockwise angle (°) with the most text detections.
    crop_bbox      : (x1, y1, x2, y2) in raw_no_text space — coordinate bridge.

    Supports tuple unpacking:
        raw_no_text, preprocessed = preprocessor.process(image)
    """
    raw_no_text: np.ndarray
    preprocessed: np.ndarray
    rotation_angle: int
    crop_bbox: tuple[int, int, int, int]

    def __iter__(self):
        yield self.raw_no_text
        yield self.preprocessed


class FloorPlanPreprocessor:
    """
    Single-image preprocessing pipeline for floor plans.

    Pipeline
    --------
    1. Multi-angle OCR  → find & erase text  → raw_no_text (BGR, uncropped)
    2. Grayscale → Gaussian blur → Otsu binarisation
    3. Crop & pad to content bounding box    → preprocessed (binary, cropped)

    Usage
    -----
    pp = FloorPlanPreprocessor()
    result = pp.process(image)      # image: BGR numpy array

    result.rotation_angle           # int — clockwise degrees for upright orientation
    result.crop_bbox                # (x1, y1, x2, y2) in raw_no_text space
    result.raw_no_text              # BGR, text removed, NOT cropped
    result.preprocessed             # binary, cropped — feed to skeletoniser

    To get the cropped color image for YOLO:
        x1, y1, x2, y2 = result.crop_bbox
        cropped_color = result.raw_no_text[y1:y2, x1:x2]
    """

    def __init__(
        self,
        confidence_threshold: float = 0.1,
        dilation_px: int = 15,
        angle_step: int = 90,
        crop_padding: int = 100,
        gaussian_ksize: int = 5,
        languages: list[str] | None = None,
        gpu: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.dilation_px = dilation_px
        self.angle_step = angle_step
        self.crop_padding = crop_padding
        self.gaussian_ksize = gaussian_ksize
        self.reader = easyocr.Reader(languages or ["en", "fr", "it"], gpu=gpu)

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, image: np.ndarray) -> PipelineResult:
        """
        Run the full pipeline on a single BGR image.

        Returns a PipelineResult that unpacks as:
            raw_no_text, preprocessed = preprocessor.process(image)
        """
        # Stage 1: text removal
        raw_no_text, best_angle = self._remove_text(image)

        # Stage 2: binarise the text-free image
        binary = self._enhance(raw_no_text, self.gaussian_ksize)

        # Stage 3: crop the binary image; raw_no_text is kept uncropped
        preprocessed, crop_bbox = self._crop_and_pad(binary, self.crop_padding)

        raw_no_text_cropped = raw_no_text[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]

        return PipelineResult(
            raw_no_text=raw_no_text_cropped,
            preprocessed=preprocessed,
            rotation_angle=best_angle,
            crop_bbox=crop_bbox,
        )

    # ── Stage 1: text removal ─────────────────────────────────────────────────

    def _remove_text(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        angle_counts: dict[int, int] = {}

        for angle in range(0, 360, self.angle_step):
            rotated    = self._rotate(image, angle)
            detections = self._detect_text(rotated)
            angle_counts[angle] = len(detections)
            if not detections:
                continue
            rot_mask  = self._build_mask(rotated, detections)
            orig_mask = self._rotate_mask_back(rot_mask, image.shape, angle)
            combined_mask = cv2.bitwise_or(combined_mask, orig_mask)

        best_angle = max(angle_counts, key=angle_counts.get)
        return self._erase(image, combined_mask), best_angle

    def _detect_text(self, img: np.ndarray) -> list[dict]:
        return [
            {"box": np.array(bbox, dtype=np.int32), "text": text, "conf": conf}
            for bbox, text, conf in self.reader.readtext(img)
            if conf >= self.confidence_threshold
        ]

    def _build_mask(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for d in detections:
            cv2.fillPoly(mask, [d["box"]], 255)
        if self.dilation_px > 0:
            side = self.dilation_px * 2 + 1
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (side, side))
            mask = cv2.dilate(mask, k)
        return mask

    # ── Stage 2: enhancement ──────────────────────────────────────────────────

    @staticmethod
    def _enhance(image: np.ndarray, gaussian_ksize: int = 5) -> np.ndarray:
        """Grayscale → Gaussian blur → Otsu binarisation."""
        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        k       = gaussian_ksize | 1  # ensure odd
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    # ── Stage 3: crop & pad ───────────────────────────────────────────────────

    @staticmethod
    def _crop_and_pad(
        gray: np.ndarray, padding: int
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        coords = cv2.findNonZero(gray)
        if coords is None:
            raise ValueError("No content found in image — cannot crop.")

        H, W = gray.shape[:2]
        x, y, w, h = cv2.boundingRect(coords)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(W, x + w + padding)
        y2 = min(H, y + h + padding)

        return gray[y1:y2, x1:x2], (x1, y1, x2, y2)

    # ── Rotation helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _rotate(image: np.ndarray, angle: int) -> np.ndarray:
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy
        return cv2.warpAffine(image, M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))

    @staticmethod
    def _rotate_mask_back(mask: np.ndarray, original_shape: tuple, angle: int) -> np.ndarray:
        h_rot, w_rot = mask.shape[:2]
        h_orig, w_orig = original_shape[:2]
        cx, cy = w_rot / 2, h_rot / 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        M[0, 2] += (w_orig / 2) - cx
        M[1, 2] += (h_orig / 2) - cy
        return cv2.warpAffine(mask, M, (w_orig, h_orig),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)

    @staticmethod
    def _erase(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = image.copy()
        out[mask == 255] = 255
        return out


if __name__ == "__main__":
    import sys

    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/PNG/raw/IIa_A01001.png"
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(img_path)

    preprocessor = FloorPlanPreprocessor(angle_step=90)
    result = preprocessor.process(image)

    out_dir = Path("../data/PNG")
    stem = Path(img_path).stem

    x1, y1, x2, y2 = result.crop_bbox
    cropped_color = result.raw_no_text[y1:y2, x1:x2]

    cv2.imwrite(str(out_dir / "no_text"      / f"{stem}.png"), result.raw_no_text)
    cv2.imwrite(str(out_dir / "crop_and_pad" / f"{stem}.png"), result.preprocessed)

    print(f"rotation_angle : {result.rotation_angle}°")
    print(f"crop_bbox      : {result.crop_bbox}")
    print(f"Saved raw_no_text  → {out_dir}/no_text/{stem}.png")
    print(f"Saved preprocessed → {out_dir}/crop_and_pad/{stem}.png")
