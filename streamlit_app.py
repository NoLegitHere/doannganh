from io import BytesIO
import pandas as pd
import streamlit as st
from PIL import Image

from app.services.image_utils import read_image_from_bytes
from app.services.runtime import get_pipeline


st.set_page_config(page_title="Nhận diện biển số xe", page_icon="🚗", layout="wide")
st.title("Hệ thống nhận diện và trích xuất thông tin biển số xe")
st.caption("Sử dụng OCR, nhận diện vùng biển số và bảng mã tỉnh/thành mới theo Thông tư 51/2025/TT-BCA.")

uploaded_file = st.file_uploader("Tải ảnh xe lên", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = read_image_from_bytes(image_bytes)
    display_image = Image.open(BytesIO(image_bytes))
    pipeline = get_pipeline()
    response = pipeline.recognize(image)
    annotated = pipeline.annotate(image, response)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh gốc")
        st.image(display_image, use_container_width=True)

    with col2:
        st.subheader("Kết quả nhận diện")
        st.image(annotated[:, :, ::-1], use_container_width=True)

    if response.predictions:
        rows = []
        for item in response.predictions:
            rows.append(
                {
                    "Biển số": item.text_formatted or item.text_normalized,
                    "Tỉnh/Thành": item.province.name if item.province else "Không xác định",
                    "Mã": item.province.code if item.province else "",
                    "Độ tin cậy": round(item.confidence, 4),
                    "Detector": round(item.detector_confidence, 4),
                    "OCR": round(item.ocr_confidence, 4),
                    "Hợp lệ": item.is_valid_vietnam_plate,
                }
            )
        st.subheader("Bảng kết quả")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.warning("Chưa phát hiện được biển số nào trong ảnh.")
else:
    st.info("Hãy tải một ảnh phương tiện để bắt đầu.")

with st.expander("Gợi ý sử dụng"):
    st.markdown("""
- Nên dùng ảnh rõ, đủ sáng, biển số không bị che khuất.
- Nếu chưa có model YOLO riêng, hệ thống sẽ fallback sang phát hiện bằng OpenCV.
- Khi có trọng số `models/license_plate_detector.pt`, hệ thống sẽ tự ưu tiên dùng mô hình đó.
""")
