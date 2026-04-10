from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.core.settings import PROJECT_ROOT, settings


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    source: str


class LicensePlateDetector:
    def __init__(self, model_path: str | None = None):
        self.requested_model_path = Path(model_path or settings.detection_model_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_path = self._resolve_model_path()
        self.model = self._load_model(self.model_path) if self.model_path is not None else None

    def _candidate_model_paths(self) -> list[Path]:
        requested = self.requested_model_path
        if not requested.is_absolute():
            requested = PROJECT_ROOT / requested

        candidates = [
            requested,
            PROJECT_ROOT / "models" / "license_plate_detector.pt",
            PROJECT_ROOT / "outputs" / "train" / "license_plate_detector" / "weights" / "best.pt",
            PROJECT_ROOT / "runs" / "detect" / "outputs" / "train" / "license_plate_detector" / "weights" / "best.pt",
            PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt",
        ]

        unique_candidates: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(candidate)
        return unique_candidates

    def _resolve_model_path(self) -> Path | None:
        for candidate in self._candidate_model_paths():
            if candidate.exists():
                return candidate
        return None

    def _load_model(self, model_path: Path):
        if model_path.exists():
            return YOLO(str(model_path))
        return None

    def _ensure_model_loaded(self) -> None:
        resolved_path = self._resolve_model_path()
        if resolved_path is None:
            return
        if self.model is None or self.model_path != resolved_path:
            self.model_path = resolved_path
            self.model = self._load_model(resolved_path)

    @property
    def active_model_path(self) -> str:
        if self.model_path is None:
            return ""
        return str(self.model_path)

    def detect(self, image: np.ndarray) -> list[Detection]:
        self._ensure_model_loaded()
        if self.model is not None:
            detections = self._detect_with_yolo(image)
            if detections:
                return detections
        return self._detect_with_opencv(image)

    def _detect_with_yolo(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(source=image, verbose=False, device=self.device)
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=conf,
                        source="yolo",
                    )
                )
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    def _detect_with_opencv(self, image: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(filtered, 30, 200)
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes: list[Detection] = []
        image_area = image.shape[0] * image.shape[1]

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:50]:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / max(h, 1)

            if area < image_area * 0.002 or area > image_area * 0.25:
                continue
            if aspect_ratio < 2.0 or aspect_ratio > 6.5:
                continue
            if len(approx) < 4 or len(approx) > 10:
                continue

            confidence = min(0.95, 0.3 + (area / image_area) * 10)
            candidate_boxes.append(
                Detection(
                    bbox=(x, y, x + w, y + h),
                    confidence=float(confidence),
                    source="opencv",
                )
            )

        unique: list[Detection] = []
        for detection in candidate_boxes:
            if not any(self._iou(detection.bbox, existing.bbox) > 0.5 for existing in unique):
                unique.append(detection)
        unique.sort(key=lambda item: item.confidence, reverse=True)
        return unique[:3]

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union == 0:
            return 0.0
        return inter_area / union
