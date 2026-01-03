import csv
import os
from pathlib import Path
from typing import Dict, List

from vision_ocr import ocr_spypoint_stamp_vision

IMAGES_DIR = Path("images")
OUT_CSV = Path("stamp_data.csv")

# If you ever want to re-run OCR and update existing rows:
UPDATE_EXISTING = os.environ.get("UPDATE_EXISTING") == "1"


def load_existing(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Returns dict keyed by filename -> row dict
    """
    if not csv_path.exists():
        return {}

    rows_by_file: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = (row.get("filename") or "").strip()
            if fn:
                rows_by_file[fn] = row
    return rows_by_file


def write_csv(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = ["filename", "date", "time", "temp_f", "temp_c", "raw_text"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    if not IMAGES_DIR.exists():
        raise SystemExit(f"Missing folder: {IMAGES_DIR.resolve()}")

    existing = load_existing(OUT_CSV)

    jpgs = sorted(IMAGES_DIR.glob("*.jpg"))
    added = 0
    updated = 0

    for p in jpgs:
        fn = p.name

        if fn in existing and not UPDATE_EXISTING:
            continue

        stamp = ocr_spypoint_stamp_vision(str(p))
        row = {
            "filename": fn,
            "date": stamp.date_mmddyyyy or "",
            "time": stamp.time_hhmm_ampm or "",
            "temp_f": "" if stamp.temp_f is None else str(stamp.temp_f),
            "temp_c": "" if stamp.temp_c is None else str(stamp.temp_c),
            "raw_text": stamp.raw_text or "",
        }

        if fn in existing:
            existing[fn] = row
            updated += 1
        else:
            existing[fn] = row
            added += 1

    # Stable ordering: by filename (your filenames encode time)
    all_rows = [existing[k] for k in sorted(existing.keys())]
    write_csv(OUT_CSV, all_rows)

    print(f"Wrote {OUT_CSV} with {len(all_rows)} total rows")
    print(f"Added {added}, updated {updated}, skipped {len(jpgs) - added - updated}")


if __name__ == "__main__":
    main()
