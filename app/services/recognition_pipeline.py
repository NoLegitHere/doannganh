import cv2
import numpy as np

from app.schemas.recognition import BoundingBox, PlatePrediction, ProvinceInfo, RecognitionResponse
from app.services.detector import LicensePlateDetector
from app.services.image_utils import crop_image
from app.services.ocr_engine import OCREngine
from app.services.plate_format import format_vietnamese_plate, is_probably_vietnamese_plate, normalize_plate_text
from app.services.province_lookup import lookup_province_by_plate


class PlateRecognitionPipeline:
    def __init__(self):
        self.detector = LicensePlateDetector()
        self.ocr = OCREngine()

    def recognize(self, image: np.ndarray) -> RecognitionResponse:
        detections = self.detector.detect(image)
        predictions: list[PlatePrediction] = []

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            plate_crop = crop_image(image, x1, y1, x2, y2)
            raw_text, ocr_conf = self.ocr.recognize_plate(plate_crop)
            normalized = normalize_plate_text(raw_text)
            formatted = format_vietnamese_plate(normalized)
            province = lookup_province_by_plate(normalized)
            predictions.append(
                PlatePrediction(
                    text_raw=raw_text,
                    text_normalized=normalized,
                    text_formatted=formatted,
                    confidence=float((detection.confidence + ocr_conf) / 2),
                    detector_confidence=float(detection.confidence),
                    ocr_confidence=float(ocr_conf),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    province=ProvinceInfo(**province) if province else None,
                    is_valid_vietnam_plate=is_probably_vietnamese_plate(normalized),
                )
            )

        predictions.sort(key=lambda item: item.confidence, reverse=True)
        height, width = image.shape[:2]
        return RecognitionResponse(
            image_width=width,
            image_height=height,
            total_plates=len(predictions),
            predictions=predictions,
        )

    def annotate(self, image: np.ndarray, response: RecognitionResponse) -> np.ndarray:
        output = image.copy()
        for prediction in response.predictions:
            label = prediction.text_formatted or prediction.text_normalized or "UNREADABLE"
            if prediction.province is not None:
                label = f"{label} | {prediction.province.name}"
            bbox = prediction.bbox
            cv2.rectangle(output, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 255, 0), 2)
            cv2.rectangle(output, (bbox.x1, max(0, bbox.y1 - 30)), (bbox.x2, bbox.y1), (0, 255, 0), -1)
            cv2.putText(output, label, (bbox.x1 + 4, max(18, bbox.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        return output
