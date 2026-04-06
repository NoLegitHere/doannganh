from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def read_image_from_path(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def encode_image_to_jpeg_bytes(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Failed to encode image to JPEG")
    return buffer.tobytes()


def crop_image(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop coordinates")
    return image[y1:y2, x1:x2].copy()


def draw_box_with_label(image: np.ndarray, box: tuple[int, int, int, int], label: str) -> np.ndarray:
    x1, y1, x2, y2 = box
    output = image.copy()
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(output, (x1, max(0, y1 - 28)), (x2, y1), (0, 255, 0), -1)
    cv2.putText(output, label, (x1 + 4, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return output
