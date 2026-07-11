from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from google.api_core.exceptions import GoogleAPICallError, RetryError
from google.cloud import vision
from PIL import Image


@dataclass(frozen=True)
class Stamp:
    date_mmddyyyy: Optional[str]
    time_hhmm_ampm: Optional[str]
    temp_f: Optional[int]
    temp_c: Optional[int]
    raw_text: str


DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
TIME_RE = re.compile(r"\b(\d{1,2}:\d{2})\s*([AP]M)\b", re.IGNORECASE)
TEMP_F_RE = re.compile(r"\b(-?\d{1,3})\s*[°º]?\s*F\b", re.IGNORECASE)
TEMP_C_RE = re.compile(r"\b(-?\d{1,3})\s*[°º]?\s*C\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _vision_client() -> vision.ImageAnnotatorClient:
    return vision.ImageAnnotatorClient()


def _normalize_date(value: str) -> Optional[str]:
    value = value.replace("-", "/")
    parts = value.split("/")
    if len(parts) != 3:
        return None
    month, day, year = parts
    if len(year) == 2:
        year = f"20{year}"
    try:
        month_i, day_i, year_i = int(month), int(day), int(year)
        if not (1 <= month_i <= 12 and 1 <= day_i <= 31 and 2000 <= year_i <= 2100):
            return None
    except ValueError:
        return None
    return f"{month_i:02d}/{day_i:02d}/{year_i:04d}"


def _parse_stamp(text: str) -> Stamp:
    clean = " ".join((text or "").split())
    date_match = DATE_RE.search(clean)
    time_match = TIME_RE.search(clean)
    temp_f_match = TEMP_F_RE.search(clean)
    temp_c_match = TEMP_C_RE.search(clean)

    date_value = _normalize_date(date_match.group(1)) if date_match else None
    time_value = (
        f"{time_match.group(1)} {time_match.group(2).upper()}" if time_match else None
    )

    temp_f = int(temp_f_match.group(1)) if temp_f_match else None
    temp_c = int(temp_c_match.group(1)) if temp_c_match else None

    # Recover the missing unit when only one temperature is recognized.
    if temp_f is None and temp_c is not None:
        temp_f = round((temp_c * 9 / 5) + 32)
    elif temp_c is None and temp_f is not None:
        temp_c = round((temp_f - 32) * 5 / 9)

    return Stamp(date_value, time_value, temp_f, temp_c, clean)


def _read_stamp_crop(image_path: str, crop_fraction: float = 0.30) -> bytes:
    path = Path(image_path)
    with Image.open(path) as image:
        image = image.convert("RGB")
        top = max(0, int(image.height * (1.0 - crop_fraction)))
        crop = image.crop((0, top, image.width, image.height))
        from io import BytesIO

        buffer = BytesIO()
        crop.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


def _detect_text(content: bytes) -> str:
    response = _vision_client().text_detection(image=vision.Image(content=content))
    if response.error.message:
        raise RuntimeError(f"Google Vision OCR failed: {response.error.message}")
    return (response.full_text_annotation.text or "").strip()


def ocr_spypoint_stamp_vision(image_path: str) -> Stamp:
    """OCR only the bottom stamp region first, then retry the full image if needed."""
    try:
        cropped_text = _detect_text(_read_stamp_crop(image_path))
        stamp = _parse_stamp(cropped_text)
        if stamp.date_mmddyyyy and stamp.time_hhmm_ampm:
            return stamp

        full_text = _detect_text(Path(image_path).read_bytes())
        full_stamp = _parse_stamp(full_text)
        return full_stamp if full_stamp.raw_text else stamp
    except (GoogleAPICallError, RetryError, OSError) as exc:
        raise RuntimeError(f"Unable to OCR {image_path}: {exc}") from exc
