import re
from typing import Optional


_NON_ALNUM_PATTERN = re.compile(r"[^A-Z0-9]")
_NUMERIC_OCR_MAP = str.maketrans({
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
})


def normalize_plate_text(text: str) -> str:
    cleaned = _NON_ALNUM_PATTERN.sub("", text.upper())
    return cleaned


def extract_province_code(text: str) -> Optional[str]:
    normalized = normalize_plate_text(text)
    if len(normalized) < 2:
        return None
    candidate = normalized[:2].translate(_NUMERIC_OCR_MAP)
    if candidate.isdigit():
        return candidate
    return None


def format_vietnamese_plate(text: str) -> str:
    normalized = normalize_plate_text(text)
    if len(normalized) < 7:
        return normalized

    prefix = normalized[:3]
    suffix = normalized[3:]

    if len(suffix) == 5:
        return f"{prefix}-{suffix[:3]}.{suffix[3:]}"
    if len(suffix) == 4:
        return f"{prefix}-{suffix[:2]}.{suffix[2:]}"
    return normalized


def is_probably_vietnamese_plate(text: str) -> bool:
    normalized = normalize_plate_text(text)
    if len(normalized) < 7:
        return False
    code = extract_province_code(normalized)
    if code is None:
        return False
    return any(ch.isalpha() for ch in normalized[2:4]) and any(ch.isdigit() for ch in normalized[3:])
