from typing import Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class ProvinceInfo(BaseModel):
    code: str
    name: str
    all_codes: list[str]


class PlatePrediction(BaseModel):
    text_raw: str = Field(default="")
    text_normalized: str = Field(default="")
    text_formatted: str = Field(default="")
    confidence: float = Field(default=0.0)
    detector_confidence: float = Field(default=0.0)
    ocr_confidence: float = Field(default=0.0)
    bbox: BoundingBox
    province: Optional[ProvinceInfo] = None
    is_valid_vietnam_plate: bool = False


class RecognitionResponse(BaseModel):
    image_width: int
    image_height: int
    total_plates: int
    predictions: list[PlatePrediction]
