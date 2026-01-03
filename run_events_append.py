import csv
import os
from pathlib import Path
from typing import Dict, List

import cv2
from megadetector.detection.run_detector import load_detector, run_detector
from speciesnet import SpeciesNetClassifier

from vision_ocr import ocr_spypoint_stamp_vision

IMAGES_DIR = Path("images")
OUT_CSV = Path("events.csv")

UPDATE_EXISTING = os.environ.get("UPDATE_EXISTING") == "1"

# thresholds
MD_ANIMAL_THRESH = 0.20
MD_PERSON_THRESH = 0.30
MD_VEHICLE_THRESH = 0.30

CAT_ANIMAL = "1"
CAT_PERSON = "2"
CAT_VEHICLE = "3"


def load_existing(csv_path: Path) -> Dict[str, Dict[str, str]]:
    if not csv_path.exists():
        return {}
    rows = {}
    with csv_path.open("r", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["filename"]] = r
    return rows


def write_csv(csv_path: Path, rows: List[Dict[str, str]]):
    fields = [
        "filename", "date", "time", "temp_f", "temp_c",
        "has_animal", "animal_conf",
        "has_person", "has_vehicle",
        "species", "species_conf", "species_top3",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def crop(img, bbox_xywh):
    h, w = img.shape[:2]
    x, y, bw, bh = bbox_xywh
    x0, y0 = int(x * w), int(y * h)
    x1, y1 = int((x + bw) * w), int((y + bh) * h)
    return img[max(0,y0):min(h,y1), max(0,x0):min(w,x1)]


def main():
    if not IMAGES_DIR.exists():
        raise SystemExit("Missing images folder")

    existing = load_existing(OUT_CSV)

    detector = load_detector("MDV6")
    sn = SpeciesNetClassifier()

    added, updated = 0, 0

    for p in sorted(IMAGES_DIR.glob("*.jpg")):
        fn = p.name
        if fn in existing and not UPDATE_EXISTING:
            continue

        stamp = ocr_spypoint_stamp_vision(str(p))
        img = cv2.imread(str(p))

        md = run_detector(detector, str(p), quiet=True)
        dets = md.get("detections", [])

        animal_dets = [d for d in dets if d["category"] == CAT_ANIMAL and d["conf"] >= MD_ANIMAL_THRESH]
        person_dets = [d for d in dets if d["category"] == CAT_PERSON and d["conf"] >= MD_PERSON_THRESH]
        vehicle_dets = [d for d in dets if d["category"] == CAT_VEHICLE and d["conf"] >= MD_VEHICLE_THRESH]

        has_animal = bool(animal_dets)
        has_person = bool(person_dets)
        has_vehicle = bool(vehicle_dets)

        animal_conf = max([d["conf"] for d in animal_dets], default=0.0)

        species = ""
        species_conf = ""
        species_top3 = ""

        if has_animal and img is not None:
            best = max(animal_dets, key=lambda d: d["conf"])
            crop_img = crop(img, best["bbox"])
            crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

            preds = sn.classify(crop_rgb, top_k=3)
            if preds:
                species = preds[0]["species"]
                species_conf = f"{preds[0]['score']:.3f}"
                species_top3 = ";".join(f"{p['species']}:{p['score']:.3f}" for p in preds)

        row = {
            "filename": fn,
            "date": stamp.date_mmddyyyy or "",
            "time": stamp.time_hhmm_ampm or "",
            "temp_f": "" if stamp.temp_f is None else str(stamp.temp_f),
            "temp_c": "" if stamp.temp_c is None else str(stamp.temp_c),
            "has_animal": str(has_animal).lower(),
            "animal_conf": f"{animal_conf:.3f}",
            "has_person": str(has_person).lower(),
            "has_vehicle": str(has_vehicle).lower(),
            "species": species,
            "species_conf": species_conf,
            "species_top3": species_top3,
        }

        existing[fn] = row
        if fn in existing:
            updated += 1
        else:
            added += 1

    all_rows = [existing[k] for k in sorted(existing.keys())]
    write_csv(OUT_CSV, all_rows)

    print(f"Wrote {OUT_CSV} with {len(all_rows)} rows")
    print(f"Added {added}, updated {updated}")


if __name__ == "__main__":
    main()
