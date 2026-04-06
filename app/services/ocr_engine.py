from functools import lru_cache

import cv2
import easyocr
import numpy as np

from app.core.settings import settings
from app.services.plate_format import normalize_plate_text


class OCREngine:
    def __init__(self):
        languages = [lang.strip() for lang in settings.default_ocr_languages.split(",") if lang.strip()]
        self.reader = get_easyocr_reader(tuple(languages), settings.easyocr_gpu)

    def recognize_plate(self, image: np.ndarray) -> tuple[str, float]:
        prepared = self._prepare(image)
        results = self.reader.readtext(prepared, detail=1, paragraph=False, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        if not results:
            return "", 0.0

        parts: list[str] = []
        confidences: list[float] = []
        for _, text, confidence in results:
            normalized = normalize_plate_text(text)
            if normalized:
                parts.append(normalized)
                confidences.append(float(confidence))

        if not parts:
            return "", 0.0

        merged = "".join(parts)
        avg_conf = float(sum(confidences) / max(len(confidences), 1))
        return merged, avg_conf

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh


@lru_cache(maxsize=4)
def get_easyocr_reader(languages: tuple[str, ...], gpu: bool):
    return easyocr.Reader(list(languages), gpu=gpu)
