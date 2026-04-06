import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.core.settings import settings
from app.services.plate_format import extract_province_code


@lru_cache(maxsize=1)
def load_province_mappings() -> dict[str, dict]:
    data = json.loads(Path(settings.province_code_path).read_text(encoding="utf-8"))
    mapping: dict[str, dict] = {}
    for item in data:
        for code in item["codes"]:
            mapping[code] = {
                "code": code,
                "name": item["name"],
                "all_codes": item["codes"],
            }
    return mapping


def lookup_province_by_code(code: str) -> Optional[dict]:
    return load_province_mappings().get(code)


def lookup_province_by_plate(plate_text: str) -> Optional[dict]:
    code = extract_province_code(plate_text)
    if code is None:
        return None
    return lookup_province_by_code(code)
