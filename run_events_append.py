import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import cv2  # still used for reading images if you want to extend later

from vision_ocr import ocr_spypoint_stamp_vision

IMAGES_DIR = Path("images")
OUT_CSV = Path("events.csv")
SPECIESNET_JSON = Path("speciesnet-results.json")

UPDATE_EXISTING = os.environ.get("UPDATE_EXISTING") == "1"

# thresholds (detections come from SpeciesNet's detector output)
ANIMAL_THRESH = 0.20
HUMAN_THRESH = 0.30
VEHICLE_THRESH = 0.30

CAT_ANIMAL = "1"
CAT_HUMAN = "2"
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


def run_speciesnet(images_dir: Path, out_json: Path):
    """
    Runs SpeciesNet on a folder and writes predictions JSON.
    Uses optional env vars for geofencing:
      SPECIESNET_COUNTRY (e.g. USA)
      SPECIESNET_ADMIN1  (e.g. TX)
    """
    cmd = [
        "python", "-m", "speciesnet.scripts.run_model",
        "--folders", str(images_dir),
        "--predictions_json", str(out_json),
    ]

    country = os.environ.get("SPECIESNET_COUNTRY", "").strip()
    admin1 = os.environ.get("SPECIESNET_ADMIN1", "").strip()

    if country:
        cmd += ["--country", country]
    if admin1:
        cmd += ["--admin1_region", admin1]

    subprocess.run(cmd, check=True)


def main():
    if not IMAGES_DIR.exists():
        raise SystemExit("Missing images folder")

    # Run SpeciesNet once for the whole folder
    run_speciesnet(IMAGES_DIR, SPECIESNET_JSON)

    with SPECIESNET_JSON.open("r") as f:
        sn = json.load(f)

    preds = sn.get("predictions", [])
    by_filename = {}
    for p in preds:
        fp = p.get("filepath", "")
        if not fp:
            continue
        by_filename[Path(fp).name] = p

    existing = load_existing(OUT_CSV)

    added, updated = 0, 0

    for img_path in sorted(IMAGES_DIR.glob("*.jpg")):
        fn = img_path.name

        if fn in existing and not UPDATE_EXISTING:
            continue

        stamp = ocr_spypoint_stamp_vision(str(img_path))

        pred = by_filename.get(fn, {})
        dets = pred.get("detections", []) or []

        animal_dets = [d for d in dets if d.get("category") == CAT_ANIMAL and float(d.get("conf", 0.0)) >= ANIMAL_THRESH]
        human_dets = [d for d in dets if d.get("category") == CAT_HUMAN and float(d.get("conf", 0.0)) >= HUMAN_THRESH]
        vehicle_dets = [d for d in dets if d.get("category") == CAT_VEHICLE and float(d.get("conf", 0.0)) >= VEHICLE_THRESH]

        has_animal = bool(animal_dets)
        has_person = bool(human_dets)
        has_vehicle = bool(vehicle_dets)

        animal_conf = max([float(d.get("conf", 0.0)) for d in animal_dets], default=0.0)

        # SpeciesNet final ensemble prediction + score
        species = pred.get("prediction", "") or ""
        species_conf_val = pred.get("prediction_score", None)
        species_conf = "" if species_conf_val is None else f"{float(species_conf_val):.3f}"

        # Top-3 from raw classifier output (if present)
        cls = pred.get("classifications", {}) or {}
        classes = cls.get("classes", []) or []
        scores = cls.get("scores", []) or []

        top3 = []
        for c, s in list(zip(classes, scores))[:3]:
            try:
                top3.append(f"{c}:{float(s):.3f}")
            except Exception:
                top3.append(f"{c}:{s}")
        species_top3 = ";".join(top3)

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

        was_existing = fn in existing
        existing[fn] = row
        if was_existing:
            updated += 1
        else:
            added += 1

    all_rows = [existing[k] for k in sorted(existing.keys())]
    write_csv(OUT_CSV, all_rows)

    print(f"Wrote {OUT_CSV} with {len(all_rows)} rows")
    print(f"Added {added}, updated {updated}")


if __name__ == "__main__":
    main()



