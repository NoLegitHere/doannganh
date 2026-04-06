# Nhận diện và trích xuất thông tin biển số xe

Dự án này là một sản phẩm hoàn chỉnh theo hướng đồ án tốt nghiệp cho bài toán nhận diện và trích xuất thông tin biển số xe tại Việt Nam. Hệ thống gồm:

- Backend `FastAPI` để nhận ảnh và trả kết quả nhận diện.
- Giao diện `Streamlit` để demo sản phẩm.
- Pipeline nhận diện biển số gồm detector + OCR + ánh xạ tỉnh/thành.
- Script tải dataset từ Kaggle, chuẩn hóa dữ liệu, huấn luyện detector YOLO và suy luận ảnh.

## Tính năng chính

- Nhận diện vùng biển số từ ảnh xe.
- Trích xuất ký tự biển số bằng OCR.
- Chuẩn hóa chuỗi biển số theo định dạng Việt Nam.
- Ánh xạ `mã tỉnh/thành -> tên tỉnh/thành` theo **bộ mã mới áp dụng cho 34 tỉnh/thành từ 01/07/2025**.
- Hỗ trợ fallback bằng OpenCV khi chưa có trọng số detector riêng.

## Cấu trúc thư mục

```text
app/
  core/
  resources/
  schemas/
  services/
configs/
scripts/
streamlit_app.py
requirements.txt
README.md
```

## 1. Tạo môi trường ảo

Trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Cấu hình môi trường

Sao chép file mẫu:

```powershell
Copy-Item .env.example .env
```

Bạn có thể chỉnh các biến như `DETECTION_MODEL_PATH`, `DEFAULT_OCR_LANGUAGES`, `EASYOCR_GPU` nếu cần.

## 3. Tải dataset từ Kaggle

Project đã chuẩn bị sẵn file `configs/kaggle_datasets.yaml` với một số dataset phù hợp cho detection và OCR.

Ví dụ:

```powershell
python scripts/download_kaggle_datasets.py
```

Hoặc chỉ tải một dataset cụ thể:

```powershell
python scripts/download_kaggle_datasets.py --only bomaich/vnlicenseplate
```

## 4. Chuẩn bị dữ liệu detection về YOLO format

```powershell
python scripts/prepare_detection_dataset.py --inputs data/raw/vnlicenseplate data/raw/car-plate-detection --output data/processed/detection
```

Sau đó tạo file cấu hình dữ liệu thực tế từ mẫu `configs/detection_data.example.yaml` nếu cần điều chỉnh đường dẫn.

## 5. Huấn luyện detector YOLO

```powershell
python scripts/train_detector.py --data configs/detection_data.example.yaml --model yolov8n.pt --epochs 80 --imgsz 640
```

Sau khi train xong, copy file `best.pt` về:

```text
models/license_plate_detector.pt
```

## 6. Chạy API

```powershell
uvicorn app.main:app --reload
```

Các endpoint chính:

- `GET /health`
- `GET /provinces`
- `POST /recognize`
- `POST /recognize/annotated`

## 7. Chạy giao diện Streamlit

```powershell
streamlit run streamlit_app.py
```

## 8. Chạy suy luận ảnh bằng CLI

```powershell
python scripts/run_inference.py path\to\image.jpg --output outputs/inference/result.jpg
```

## Nguồn mã tỉnh/thành

Bảng mã trong `app/resources/province_codes_2025.json` đang dùng **bộ mã tỉnh/thành mới** theo bài tra cứu cập nhật dẫn chiếu **Thông tư 51/2025/TT-BCA**, áp dụng cho 34 tỉnh/thành từ ngày 01/07/2025.

## Ghi chú kỹ thuật

- Nếu chưa có model YOLO riêng, hệ thống dùng phát hiện heuristic bằng OpenCV để đảm bảo vẫn demo được toàn pipeline.
- OCR hiện dùng `EasyOCR` để rút ngắn thời gian xây dựng sản phẩm.
- Với dữ liệu tốt hơn, bạn có thể thay OCR bằng CRNN hoặc PaddleOCR để tăng độ chính xác.
- Với video thời gian thực, có thể mở rộng thêm pipeline đọc frame và tracking phương tiện.
