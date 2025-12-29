import re
from dataclasses import dataclass
from typing import Optional

from google.cloud import vision


@dataclass
class Stamp:
    date_mmddyyyy: Optional[str]
    time_hhmm_ampm: Optional[str]
    temp_f: Optional[int]
    temp_c: Optional[int]
    raw_text: str


DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
TIME_RE = re.compile(r"\b(\d{1,2}:\d{2})\s*([AP]M)\b", re.IGNORECASE)
TEMP_F_RE = re.compile(r"\b(-?\d{1,3})\s*°?\s*F\b", re.IGNORECASE)
TEMP_C_RE = re.compile(r"\b(-?\d{1,3})\s*°?\s*C\b", re.IGNORECASE)


def ocr_spypoint_stamp_vision(image_path: str) -> Stamp:
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as f:
        content = f.read()

    image = vision.Image(content=content)
    resp = client.text_detection(image=image)

    text = (resp.full_text_annotation.text or "").strip()
    clean = " ".join(text.split())

    date = DATE_RE.search(clean)
    time = TIME_RE.search(clean)
    tf = TEMP_F_RE.search(clean)
    tc = TEMP_C_RE.search(clean)

    return Stamp(
        date_mmddyyyy=date.group(1) if date else None,
        time_hhmm_ampm=f"{time.group(1)} {time.group(2).upper()}" if time else None,
        temp_f=int(tf.group(1)) if tf else None,
        temp_c=int(tc.group(1)) if tc else None,
        raw_text=clean,
    )
