from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from app.schemas.recognition import RecognitionResponse
from app.services.image_utils import read_image_from_bytes
from app.services.province_lookup import load_province_mappings
from app.services.runtime import get_pipeline


app = FastAPI(title="Vietnamese License Plate Recognition API", version="1.0.0")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/provinces")
def get_provinces() -> list[dict]:
    grouped: dict[str, dict] = {}
    for item in load_province_mappings().values():
        grouped[item["name"]] = item
    return sorted(grouped.values(), key=lambda value: value["name"])


@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_image(file: UploadFile = File(...)) -> RecognitionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    image = read_image_from_bytes(image_bytes)
    return get_pipeline().recognize(image)


@app.post("/recognize/annotated")
async def recognize_annotated_image(file: UploadFile = File(...)) -> StreamingResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    image = read_image_from_bytes(image_bytes)
    pipeline = get_pipeline()
    response = pipeline.recognize(image)
    annotated = pipeline.annotate(image, response)
    annotated_rgb = annotated[:, :, ::-1]
    pil_image = Image.fromarray(annotated_rgb)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")
