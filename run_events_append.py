import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from species_normalization import normalize_species, normalize_event_type
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

# your rule
SPECIES_PICK_CONF = 0.60

FIELDS = [
    "camera",
    "filename",
    "date",
    "time",
    "temp_f",
    "temp_c",

    "event_type",
    "animal_conf",
    "human_conf",
    "vehicle_conf",

    "species",
    "species_clean",
    "species_conf",

    "top1_species",
    "top1_conf",
    "top2_species",
    "top2_conf",
    "top3_species",
    "top3_conf",
]


def last_after_semicolon(label: str) -> str:
    if not label:
        return ""
    s = str(label).strip()
    last = s.rsplit(";", 1)[-1].strip()
    return last.replace("_", " ")


def row_key(camera: str, filename: str) -> str:
    return f"{camera}::{filename}"


def load_existing(csv_path: Path) -> Dict[str, Dict[str, str]]:
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
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k, "") or "") for k in FIELDS})


def run_speciesnet(images_dir: Path, out_json: Path):
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


def _to_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def choose_best_species_label(
    event_type: str,
    species: str,
    species_conf: Optional[float],
    top3: List[Tuple[str, str]],
) -> Tuple[str, Optional[float]]:
    """
    Your rule:
    - If species OR top1/top2/top3 has confidence >= 0.60, use the best one.
    - Otherwise keep species as-is (it will likely normalize to a broad class or Other).
    """
    if event_type != "animal":
        return event_type, None

    candidates: List[Tuple[str, float]] = []

    if species and species_conf is not None:
        candidates.append((species, species_conf))

    for (lab, conf_s) in top3[:3]:
        c = _to_float(conf_s)
        if lab and c is not None:
            candidates.append((lab, c))

    # Only accept confident candidates
    confident = [(lab, c) for (lab, c) in candidates if c >= SPECIES_PICK_CONF]

    if confident:
        confident.sort(key=lambda x: x[1], reverse=True)
        return confident[0][0], confident[0][1]

    # fallback: original prediction label
    return species, species_conf


def main():
    if not IMAGES_DIR.exists():
        raise SystemExit("Missing images folder")

    run_speciesnet(IMAGES_DIR, SPECIESNET_JSON)

    with SPECIESNET_JSON.open("r") as f:
        sn = json.load(f)

    preds = sn.get("predictions", []) or []

    by_relpath: Dict[str, dict] = {}
    for p in preds:
        fp = p.get("filepath", "") or ""
        if not fp:
            continue
        try:
            rel = Path(fp).resolve().relative_to(IMAGES_DIR.resolve())
            by_relpath[str(rel)] = p
        except Exception:
            # last resort
            by_relpath[Path(fp).name] = p

    existing = load_existing(OUT_CSV)
    added, updated = 0, 0

    for img_path in sorted(IMAGES_DIR.rglob("*.jpg")):
        fn = img_path.name
        camera = img_path.parent.name if img_path.parent != IMAGES_DIR else "unknown"
        key = row_key(camera, fn)

        if key in existing and not UPDATE_EXISTING:
            continue

        stamp = ocr_spypoint_stamp_vision(str(img_path))

        rel_key = str(img_path.resolve().relative_to(IMAGES_DIR.resolve()))
        pred = by_relpath.get(rel_key, {})
        dets = pred.get("detections", []) or []

        animal_conf = max_conf_for_category(dets, CAT_ANIMAL)
        human_conf = max_conf_for_category(dets, CAT_HUMAN)
        vehicle_conf = max_conf_for_category(dets, CAT_VEHICLE)

        event_type = pick_event_type(animal_conf, human_conf, vehicle_conf)

        # Base predicted label + score
        raw_species = pred.get("prediction", "") or ""
        species = last_after_semicolon(raw_species)

        score_val = pred.get("prediction_score", None)
        species_conf_f: Optional[float] = None
        if score_val is not None:
            try:
                species_conf_f = float(score_val)
            except Exception:
                species_conf_f = None

        # Top3
        top3 = extract_top3(pred)
        while len(top3) < 3:
            top3.append(("", ""))

        # If non-animal, overwrite labels consistently
        if event_type in ("human", "vehicle"):
            species = event_type
            species_conf_f = human_conf if event_type == "human" else vehicle_conf
            top3 = [("", ""), ("", ""), ("", "")]
        elif event_type == "blank":
            species = ""
            species_conf_f = None
            top3 = [("", ""), ("", ""), ("", "")]

        # Choose best label using your 0.60 rule (animal only)
        chosen_label, chosen_conf = choose_best_species_label(event_type, species, species_conf_f, top3)

        # Normalize to broad, stable categories
        if event_type == "animal":
            species_clean = normalize_species(chosen_label)
        elif event_type == "human":
            species_clean = "Human"
        elif event_type == "vehicle":
            species_clean = "Vehicle"
        else:
            species_clean = "Other"

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

            # keep raw species (human/vehicle/blank overwritten above)
            "species": species,
            "species_clean": species_clean,

            # keep original species_conf (for debugging) + also chosen conf if you want
            "species_conf": "" if species_conf_f is None else f"{species_conf_f:.3f}",

            "top1_species": top3[0][0],
            "top1_conf": top3[0][1],
            "top2_species": top3[1][0],
            "top2_conf": top3[1][1],
            "top3_species": top3[2][0],
            "top3_conf": top3[2][1],
        }

        was_existing = key in existing
        existing[key] = row
        updated += 1 if was_existing else 0
        added += 0 if was_existing else 1

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
