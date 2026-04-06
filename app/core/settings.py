from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Vietnamese License Plate Recognition"
    app_env: str = "development"
    log_level: str = "INFO"

    models_dir: str = str(PROJECT_ROOT / "models")
    output_dir: str = str(PROJECT_ROOT / "outputs")
    detection_model_path: str = str(PROJECT_ROOT / "models" / "license_plate_detector.pt")
    province_code_path: str = str(PROJECT_ROOT / "app" / "resources" / "province_codes_2025.json")
    default_ocr_languages: str = "en"
    easyocr_gpu: bool = False
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
