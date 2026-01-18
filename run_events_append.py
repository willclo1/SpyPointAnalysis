import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

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
    # ✅ new: camera name
    "camera",

    "filename",
    "date",
    "time",
    "temp_f",
    "temp_c",

    # clearer event summary
    "event_type",          # animal / human / vehicle / blank
    "animal_conf",
    "human_conf",
    "vehicle_conf",

    # species prediction (only meaningful for animals, otherwise repeats event_type)
    "species",
    "species_conf",

    # top-3 broken out for readability
    "top1_species",
    "top1_conf",
    "top2_species",
    "top2_conf",
    "top3_species",
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
    """
    Unique key per event row. Prevents collisions when different cameras
    have the same filename.
    """
    return f"{camera}::{filename}"


def load_existing(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Loads existing rows keyed by (camera, filename).
    Gracefully handles old schema that may not have 'camera'.
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

    # ✅ Run SpeciesNet once for the whole folder (it will see subfolders too)
    run_speciesnet(IMAGES_DIR, SPECIESNET_JSON)

    with SPECIESNET_JSON.open("r") as f:
        sn = json.load(f)

    preds = sn.get("predictions", []) or []

    # ✅ Map predictions by relative path from IMAGES_DIR to avoid filename collisions
    # Example key: "Cam1/IMG_123.jpg"
    by_relpath: Dict[str, dict] = {}
    for p in preds:
        fp = p.get("filepath", "") or ""
        if not fp:
            continue
        try:
            rel = Path(fp).resolve().relative_to(IMAGES_DIR.resolve())
        except Exception:
            # Fallback: try best-effort key by name if paths are weird
            rel = Path(fp).name
        by_relpath[str(rel)] = p

    existing = load_existing(OUT_CSV)
    added, updated = 0, 0

    # ✅ Process recursively: images/<camera>/<file>.jpg
    for img_path in sorted(IMAGES_DIR.rglob("*.jpg")):
        fn = img_path.name

        # Camera name = immediate parent folder of the image
        # images/CamA/photo.jpg -> camera="CamA"
        camera = img_path.parent.name if img_path.parent != IMAGES_DIR else "unknown"

        key = row_key(camera, fn)

        if key in existing and not UPDATE_EXISTING:
            continue

        stamp = ocr_spypoint_stamp_vision(str(img_path))

        # Match SpeciesNet by relative path
        rel_key = str(img_path.resolve().relative_to(IMAGES_DIR.resolve()))
        pred = by_relpath.get(rel_key, {})
        dets = pred.get("detections", []) or []

        animal_conf = max_conf_for_category(dets, CAT_ANIMAL)
        human_conf = max_conf_for_category(dets, CAT_HUMAN)
        vehicle_conf = max_conf_for_category(dets, CAT_VEHICLE)

        event_type = pick_event_type(animal_conf, human_conf, vehicle_conf)

        raw_species = pred.get("prediction", "") or ""
        species = last_after_semicolon(raw_species)

        score_val = pred.get("prediction_score", None)
        species_conf = "" if score_val is None else f"{float(score_val):.3f}"

        top3 = extract_top3(pred)
        while len(top3) < 3:
            top3.append(("", ""))

        if event_type in ("human", "vehicle"):
            species = event_type
            species_conf = f"{(human_conf if event_type == 'human' else vehicle_conf):.3f}"
            top3 = [("", ""), ("", ""), ("", "")]
        elif event_type == "blank":
            species = ""
            species_conf = ""
            top3 = [("", ""), ("", ""), ("", "")]
        else:
            if not species and top3[0][0]:
                species = top3[0][0]
                species_conf = top3[0][1]

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
            "species_conf": species_conf,

            "top1_species": top3[0][0],
            "top1_conf": top3[0][1],
            "top2_species": top3[1][0],
            "top2_conf": top3[1][1],
            "top3_species": top3[2][0],
            "top3_conf": top3[2][1],
        }

        was_existing = key in existing
        existing[key] = row
        if was_existing:
            updated += 1
        else:
            added += 1

    # Stable ordering: camera then filename (then whatever else)
    def _sort_key(k: str):
        cam, fn = k.split("::", 1)
        return (cam, fn)

    all_rows = [existing[k] for k in sorted(existing.keys(), key=_sort_key)]

    write_table(OUT_CSV, all_rows, delimiter=",")
    write_table(OUT_TSV, all_rows, delimiter="\t")

    print(f"Wrote {OUT_CSV} and {OUT_TSV} with {len(all_rows)} rows")
    print(f"Added {added}, updated {updated}")


if __name__ == "__main__":
    main()
