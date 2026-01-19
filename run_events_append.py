# run_events_append.py
#
# Builds / updates events.csv + events.tsv from images/<camera>/*.jpg
# - Runs SpeciesNet over all images (including camera subfolders)
# - OCRs the Spypoint stamp for date/time/temp
# - Writes a clean "species_clean" field using species_normalization.normalize_species
# - Keeps camera in a dedicated column
# - Keys rows by camera::filename so multiple cameras can share filenames safely
#
# Expected folder layout:
#   images/
#     gate/
#       PICT1234.jpg
#     feeder/
#       PICT5678.jpg
#     ravine/
#       ...
#
# Env:
#   UPDATE_EXISTING=1   -> recompute rows even if already present
#   SPECIESNET_COUNTRY=USA
#   SPECIESNET_ADMIN1=TX   (optional)

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from species_normalization import normalize_species
from vision_ocr import ocr_spypoint_stamp_vision

IMAGES_DIR = Path("images")
OUT_CSV = Path("events.csv")
OUT_TSV = Path("events.tsv")
SPECIESNET_JSON = Path("speciesnet-results.json")

UPDATE_EXISTING = os.environ.get("UPDATE_EXISTING") == "1"

# thresholds (detections come from SpeciesNet output)
ANIMAL_THRESH = 0.20
HUMAN_THRESH = 0.30
VEHICLE_THRESH = 0.30

CAT_ANIMAL = "1"
CAT_HUMAN = "2"
CAT_VEHICLE = "3"

FIELDS = [
    # ✅ camera identity
    "camera",

    # file identity
    "filename",

    # OCR stamp
    "date",
    "time",
    "temp_f",
    "temp_c",

    # event summary
    "event_type",  # animal / human / vehicle / blank
    "animal_conf",
    "human_conf",
    "vehicle_conf",

    # species prediction
    "species",         # raw-ish label
    "species_clean",   # ✅ normalized label for charts/filters
    "species_conf",

    # top-3
    "top1_species",
    "top1_species_clean",
    "top1_conf",
    "top2_species",
    "top2_species_clean",
    "top2_conf",
    "top3_species",
    "top3_species_clean",
    "top3_conf",
]


def last_after_semicolon(label: str) -> str:
    """
    If label is like 'id;tax;tax;white_tailed_deer', returns 'white tailed deer'.
    If no semicolon, returns label as-is.
    """
    if not label:
        return ""
    s = str(label).strip()
    last = s.rsplit(";", 1)[-1].strip()
    return last.replace("_", " ")


def row_key(camera: str, filename: str) -> str:
    """Unique key per event row to avoid filename collisions across cameras."""
    return f"{camera}::{filename}"


def load_existing(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Loads existing rows keyed by (camera, filename).
    Gracefully handles old schema that may not have 'camera' or 'species_clean'.
    """
    if not csv_path.exists():
        return {}

    rows: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fn = (r.get("filename") or "").strip()
            cam = (r.get("camera") or "").strip() or "unknown"
            if not fn:
                continue

            normalized = {k: (r.get(k, "") or "") for k in FIELDS}
            # If older file didn't have species_clean, derive it now
            if not normalized.get("species_clean"):
                normalized["species_clean"] = normalize_species(normalized.get("species", ""))
            # If top*_clean missing, derive as well
            for i in (1, 2, 3):
                raw = normalized.get(f"top{i}_species", "")
                key_clean = f"top{i}_species_clean"
                if key_clean in normalized and not normalized.get(key_clean):
                    normalized[key_clean] = normalize_species(raw)

            rows[row_key(cam, fn)] = normalized
    return rows


def write_table(path: Path, rows: List[Dict[str, str]], delimiter: str):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=FIELDS,
            delimiter=delimiter,
            quoting=csv.QUOTE_MINIMAL,
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k, "") or "") for k in FIELDS})


def run_speciesnet(images_dir: Path, out_json: Path):
    """
    Runs SpeciesNet on a folder and writes predictions JSON.

    Optional env vars (geofencing):
      SPECIESNET_COUNTRY (e.g. USA)
      SPECIESNET_ADMIN1  (e.g. TX)
    """
    cmd = [
        "python",
        "-m",
        "speciesnet.scripts.run_model",
        "--folders",
        str(images_dir),
        "--predictions_json",
        str(out_json),
    ]

    country = os.environ.get("SPECIESNET_COUNTRY", "").strip()
    admin1 = os.environ.get("SPECIESNET_ADMIN1", "").strip()

    if country:
        cmd += ["--country", country]
    if admin1:
        cmd += ["--admin1_region", admin1]

    subprocess.run(cmd, check=True)


def max_conf_for_category(dets: List[dict], category: str) -> float:
    vals = [float(d.get("conf", 0.0)) for d in dets if d.get("category") == category]
    return max(vals, default=0.0)


def pick_event_type(animal_c: float, human_c: float, vehicle_c: float) -> str:
    a_ok = animal_c >= ANIMAL_THRESH
    h_ok = human_c >= HUMAN_THRESH
    v_ok = vehicle_c >= VEHICLE_THRESH

    candidates: List[Tuple[str, float]] = []
    if a_ok:
        candidates.append(("animal", animal_c))
    if h_ok:
        candidates.append(("human", human_c))
    if v_ok:
        candidates.append(("vehicle", vehicle_c))

    if not candidates:
        return "blank"

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def extract_top3(pred: dict) -> List[Tuple[str, str]]:
    """
    Returns up to 3 (label, conf_string) tuples from SpeciesNet classifications, if present.
    Converts 'a;b;c;white_tailed_deer' -> 'white tailed deer'
    """
    cls = pred.get("classifications", {}) or {}
    classes = cls.get("classes", []) or []
    scores = cls.get("scores", []) or []

    out: List[Tuple[str, str]] = []
    for c, s in list(zip(classes, scores))[:3]:
        label = last_after_semicolon(c)
        try:
            out.append((label, f"{float(s):.3f}"))
        except Exception:
            out.append((label, str(s)))
    return out


def main():
    if not IMAGES_DIR.exists():
        raise SystemExit("Missing images folder")

    # Run SpeciesNet once for the whole folder (it will see subfolders too)
    run_speciesnet(IMAGES_DIR, SPECIESNET_JSON)

    with SPECIESNET_JSON.open("r") as f:
        sn = json.load(f)

    preds = sn.get("predictions", []) or []

    # Map predictions by relative path from IMAGES_DIR to avoid filename collisions
    # Example key: "gate/IMG_123.jpg"
    by_relpath: Dict[str, dict] = {}
    for p in preds:
        fp = p.get("filepath", "") or ""
        if not fp:
            continue
        try:
            rel = Path(fp).resolve().relative_to(IMAGES_DIR.resolve())
            by_relpath[str(rel)] = p
        except Exception:
            # Fallback: if paths are weird, at least store by basename
            by_relpath[Path(fp).name] = p

    existing = load_existing(OUT_CSV)
    added, updated = 0, 0

    # Process recursively: images/<camera>/<file>.jpg
    for img_path in sorted(IMAGES_DIR.rglob("*.jpg")):
        fn = img_path.name

        camera = img_path.parent.name if img_path.parent != IMAGES_DIR else "unknown"
        key = row_key(camera, fn)

        if key in existing and not UPDATE_EXISTING:
            continue

        # OCR stamp
        stamp = ocr_spypoint_stamp_vision(str(img_path))

        # Match SpeciesNet by relative path
        pred = {}
        try:
            rel_key = str(img_path.resolve().relative_to(IMAGES_DIR.resolve()))
            pred = by_relpath.get(rel_key, {})
        except Exception:
            pred = by_relpath.get(fn, {})

        dets = pred.get("detections", []) or []
        animal_conf = max_conf_for_category(dets, CAT_ANIMAL)
        human_conf = max_conf_for_category(dets, CAT_HUMAN)
        vehicle_conf = max_conf_for_category(dets, CAT_VEHICLE)

        event_type = pick_event_type(animal_conf, human_conf, vehicle_conf)

        # Primary prediction (animals)
        raw_species = pred.get("prediction", "") or ""
        species = last_after_semicolon(raw_species)

        score_val = pred.get("prediction_score", None)
        species_conf = "" if score_val is None else f"{float(score_val):.3f}"

        # Top 3 classes (animals)
        top3 = extract_top3(pred)
        while len(top3) < 3:
            top3.append(("", ""))

        # Defaults for *_clean
        species_clean = normalize_species(species)
        top1_clean = normalize_species(top3[0][0])
        top2_clean = normalize_species(top3[1][0])
        top3_clean = normalize_species(top3[2][0])

        # If it’s not an animal event, force human-friendly labels
        if event_type in ("human", "vehicle"):
            species = event_type
            species_clean = "Person" if event_type == "human" else "Vehicle"
            species_conf = f"{(human_conf if event_type == 'human' else vehicle_conf):.3f}"

            top3 = [("", ""), ("", ""), ("", "")]
            top1_clean = top2_clean = top3_clean = ""

        elif event_type == "blank":
            species = ""
            species_clean = "Other"
            species_conf = ""
            top3 = [("", ""), ("", ""), ("", "")]
            top1_clean = top2_clean = top3_clean = ""

        else:
            # animal event but prediction empty: fallback to top1 if available
            if not species and top3[0][0]:
                species = top3[0][0]
                species_conf = top3[0][1]
                species_clean = normalize_species(species)

        row = {
            "camera": camera,

            "filename": fn,
            "date": stamp.date_mmddyyyy or "",
            "time": stamp.time_hhmm_ampm or "",
            "temp_f": "" if stamp.temp_f is None else str(stamp.temp_f),
            "temp_c": "" if stamp.temp_c is None else str(stamp.temp_c),

            "event_type": event_type,
            "animal_conf": f"{animal_conf:.3f}",
            "human_conf": f"{human_conf:.3f}",
            "vehicle_conf": f"{vehicle_conf:.3f}",

            "species": species,
            "species_clean": species_clean,
            "species_conf": species_conf,

            "top1_species": top3[0][0],
            "top1_species_clean": top1_clean,
            "top1_conf": top3[0][1],

            "top2_species": top3[1][0],
            "top2_species_clean": top2_clean,
            "top2_conf": top3[1][1],

            "top3_species": top3[2][0],
            "top3_species_clean": top3_clean,
            "top3_conf": top3[2][1],
        }

        was_existing = key in existing
        existing[key] = row
        if was_existing:
            updated += 1
        else:
            added += 1

    # Stable ordering: camera then filename
    def _sort_key(k: str):
        cam, fn = k.split("::", 1)
        return (cam, fn)

    all_rows = [existing[k] for k in sorted(existing.keys(), key=_sort_key)]

    # Write both formats
    write_table(OUT_CSV, all_rows, delimiter=",")
    write_table(OUT_TSV, all_rows, delimiter="\t")

    print(f"Wrote {OUT_CSV} and {OUT_TSV} with {len(all_rows)} rows")
    print(f"Added {added}, updated {updated}")


if __name__ == "__main__":
    main()
