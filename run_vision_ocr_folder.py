import csv
from pathlib import Path

from vision_ocr import ocr_spypoint_stamp_vision


IMAGES_DIR = Path("images")
OUT_CSV = Path("stamp_data.csv")


def main():
    if not IMAGES_DIR.exists():
        raise SystemExit(f"Missing folder: {IMAGES_DIR.resolve()}")

    rows = []
    for p in sorted(IMAGES_DIR.glob("*.jpg")):
        stamp = ocr_spypoint_stamp_vision(str(p))
        rows.append({
            "filename": p.name,
            "date": stamp.date_mmddyyyy,
            "time": stamp.time_hhmm_ampm,
            "temp_f": stamp.temp_f,
            "temp_c": stamp.temp_c,
            "raw_text": stamp.raw_text,
        })

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "date", "time", "temp_f", "temp_c", "raw_text"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {OUT_CSV} with {len(rows)} rows")


if __name__ == "__main__":
    main()
