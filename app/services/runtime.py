from functools import lru_cache

from app.services.recognition_pipeline import PlateRecognitionPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> PlateRecognitionPipeline:
    return PlateRecognitionPipeline()
