from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.schemas.recognition import RecognitionResponse
from app.services.image_utils import read_image_from_bytes
from app.services.province_lookup import load_province_mappings
from app.services.runtime import get_pipeline


OLD_PROVINCE_MAP = {
    "11": "Cao Bằng", "12": "Lạng Sơn", "14": "Quảng Ninh", "15": "Hải Phòng", "16": "Hải Phòng",
    "17": "Thái Bình", "18": "Nam Định", "19": "Phú Thọ", "20": "Thái Nguyên", "21": "Yên Bái",
    "22": "Tuyên Quang", "23": "Hà Giang", "24": "Lào Cai", "25": "Lai Châu", "26": "Sơn La",
    "27": "Điện Biên", "28": "Hòa Bình", "29": "Hà Nội", "30": "Hà Nội", "31": "Hà Nội",
    "32": "Hà Nội", "33": "Hà Nội", "40": "Hà Nội", "34": "Hải Dương", "35": "Ninh Bình",
    "36": "Thanh Hóa", "37": "Nghệ An", "38": "Hà Tĩnh", "39": "Đồng Nai", "41": "TP. Hồ Chí Minh",
    "43": "Đà Nẵng", "47": "Đắk Lắk", "48": "Đắk Nông", "49": "Lâm Đồng", "50": "TP. Hồ Chí Minh",
    "51": "TP. Hồ Chí Minh", "52": "TP. Hồ Chí Minh", "53": "TP. Hồ Chí Minh", "54": "TP. Hồ Chí Minh",
    "55": "TP. Hồ Chí Minh", "56": "TP. Hồ Chí Minh", "57": "TP. Hồ Chí Minh", "58": "TP. Hồ Chí Minh",
    "59": "TP. Hồ Chí Minh", "60": "Đồng Nai", "61": "Bình Dương", "62": "Long An", "63": "Tiền Giang",
    "64": "Vĩnh Long", "65": "Cần Thơ", "66": "Đồng Tháp", "67": "An Giang", "68": "Kiên Giang",
    "69": "Cà Mau", "70": "Tây Ninh", "71": "Bến Tre", "72": "Bà Rịa - Vũng Tàu", "73": "Quảng Bình",
    "74": "Quảng Trị", "75": "Thừa Thiên Huế", "76": "Quảng Ngãi", "77": "Bình Định", "78": "Phú Yên",
    "79": "Khánh Hòa", "80": "Cục Cảnh sát giao thông", "81": "Gia Lai", "82": "Kon Tum", "83": "Sóc Trăng",
    "84": "Trà Vinh", "85": "Ninh Thuận", "86": "Bình Thuận", "88": "Vĩnh Phúc", "89": "Hưng Yên",
    "90": "Hà Nam", "92": "Quảng Nam", "93": "Bình Phước", "94": "Bạc Liêu", "95": "Hậu Giang",
    "97": "Bắc Kạn", "98": "Bắc Giang", "99": "Bắc Ninh"
}

app = FastAPI(title="Vietnamese License Plate Recognition API", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    image = read_image_from_bytes(image_bytes)
    response = get_pipeline().recognize(image)
    
    results = []
    if response.predictions:
        for item in response.predictions:
            if not item.is_valid_vietnam_plate:
                continue
            code = item.province.code if item.province else ""
            old_city = OLD_PROVINCE_MAP.get(code, "Không xác định") if code else "Không xác định"
            results.append({
                "plate": item.text_formatted or item.text_normalized,
                "city": item.province.name if item.province else "Không xác định",
                "old_city": old_city
            })
            
    return {"success": True, "results": results}


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
