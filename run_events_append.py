# edit_run_events_append.py
import csv
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from species_normalization import normalize_species
from vision_ocr import ocr_spypoint_stamp_vision
from moon import moon_info  # <-- add this file (moon.py) as I sent earlier

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

# Species selection thresholds
PRIMARY_SPECIES_MIN = float(os.environ.get("SPECIES_STRONG_THRESH", "0.60"))  # allow override
SECONDARY_MIN = 0.35  # fallback so we don't throw everything to Other/Unknown

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

    # raw best label + confidence
    "species",
    "species_conf",

    # cleaned + grouped for charts
    "species_clean",
    "species_group",

    # moon data
    "moon_phase",
    "moon_illumination",
    "moon_age_days",

    # top-3 broken out (raw-ish, but readable)
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


def _to_float(x: Optional[object]) -> float:
    try:
        if x is None:
            return 0.0
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return 0.0
        return float(s)
    except Exception:
        return 0.0


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
        sys.executable, "-m", "speciesnet.scripts.run_model",
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


def extract_top3(pred: dict) -> List[Tuple[str, float]]:
    """
    Returns up to 3 (label, score) from SpeciesNet classifications.
    """
    cls = pred.get("classifications", {}) or {}
    classes = cls.get("classes", []) or []
    scores = cls.get("scores", []) or []

    out: List[Tuple[str, float]] = []
    for c, s in list(zip(classes, scores))[:3]:
        label = last_after_semicolon(c)
        out.append((label, _to_float(s)))
    while len(out) < 3:
        out.append(("", 0.0))
    return out


def _is_candidate_usable(label: str) -> bool:
    """
    Block junk/broad labels at selection-time so we don't "choose" bird/animal/no cv result.
    normalize_species will also guard, but we want to avoid choosing garbage in the first place.
    """
    low = (label or "").strip().lower()
    if not low:
        return False
    if low in ("blank", "no cv result", "animal", "bird", "corvus species", "canis species"):
        return False
    if "no cv" in low:
        return False
    return True


def choose_best_species_label(pred: dict) -> Tuple[str, float]:
    """
    Choose best label using:
    - candidates = prediction + top1/top2/top3
    - if any usable candidate >= PRIMARY_SPECIES_MIN, pick the highest among those
    - else pick best usable candidate >= SECONDARY_MIN
    - else ("", 0.0)
    """
    candidates: List[Tuple[str, float]] = []

    raw_pred = last_after_semicolon(pred.get("prediction", "") or "")
    pred_score = _to_float(pred.get("prediction_score", None))
    if raw_pred:
        candidates.append((raw_pred, pred_score))

    top3 = extract_top3(pred)
    for lab, sc in top3:
        if lab:
            candidates.append((lab, sc))

    usable = [(l, s) for (l, s) in candidates if _is_candidate_usable(l)]
    if not usable:
        return ("", 0.0)

    strong = [(l, s) for (l, s) in usable if s >= PRIMARY_SPECIES_MIN]
    if strong:
        strong.sort(key=lambda x: x[1], reverse=True)
        return strong[0]

    weak = [(l, s) for (l, s) in usable if s >= SECONDARY_MIN]
    if weak:
        weak.sort(key=lambda x: x[1], reverse=True)
        return weak[0]

    return ("", 0.0)


def parse_stamp_datetime(date_mmddyyyy: str, time_hhmm_ampm: str) -> Optional[datetime]:
    d = (date_mmddyyyy or "").strip()
    t = (time_hhmm_ampm or "").strip()
    if not d or not t:
        return None
    try:
        return datetime.strptime(f"{d} {t}", "%m/%d/%Y %I:%M %p")
    except Exception:
        return None


def compute_moon_fields(stamp) -> Tuple[str, str, str]:
    """
    Returns (moon_phase, moon_illumination, moon_age_days)
    """
    dt = parse_stamp_datetime(stamp.date_mmddyyyy or "", stamp.time_hhmm_ampm or "")
    if not dt:
        return ("", "", "")
    mi = moon_info(dt)
    return (mi.phase_name, f"{mi.illumination:.3f}", f"{mi.age_days:.2f}")


def main():
    if not IMAGES_DIR.exists():
        raise SystemExit("Missing images folder")

    # Run SpeciesNet once for the whole folder
    run_speciesnet(IMAGES_DIR, SPECIESNET_JSON)

    with SPECIESNET_JSON.open("r") as f:
        sn = json.load(f)

    preds = sn.get("predictions", []) or []

    # Map predictions by relative path from IMAGES_DIR
    by_relpath: Dict[str, dict] = {}
    for p in preds:
        fp = p.get("filepath", "") or ""
        if not fp:
            continue
        try:
            rel = Path(fp).resolve().relative_to(IMAGES_DIR.resolve())
            by_relpath[str(rel)] = p
        except Exception:
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
        moon_phase, moon_illum, moon_age = compute_moon_fields(stamp)

        rel_key = str(img_path.resolve().relative_to(IMAGES_DIR.resolve()))
        pred = by_relpath.get(rel_key, {})
        dets = pred.get("detections", []) or []

        animal_conf = max_conf_for_category(dets, CAT_ANIMAL)
        human_conf = max_conf_for_category(dets, CAT_HUMAN)
        vehicle_conf = max_conf_for_category(dets, CAT_VEHICLE)

        event_type = pick_event_type(animal_conf, human_conf, vehicle_conf)
        event_type = event_type

        # Top-3 (raw labels, readable)
        top3 = extract_top3(pred)
        top1_label, top1_score = top3[0]
        top2_label, top2_score = top3[1]
        top3_label, top3_score = top3[2]

        # Decide species fields
        if event_type == "human":
            species = "human"
            species_conf = f"{human_conf:.3f}"
            species_clean, species_group = ("Human", "Human")

        elif event_type == "vehicle":
            species = "vehicle"
            species_conf = f"{vehicle_conf:.3f}"
            species_clean, species_group = ("Vehicle", "Vehicle")

        elif event_type == "blank":
            species = ""
            species_conf = ""
            species_clean, species_group = ("", "")

        else:
            # animal: choose best label using confidence logic
            best_label, best_score = choose_best_species_label(pred)

            species = best_label
            species_conf = "" if best_score <= 0 else f"{best_score:.3f}"

            # normalize into canonical + grouping
            species_clean, species_group = normalize_species(best_label)

            # if still unknown, keep it as Unknown (don’t silently become “Other”)
            if not species_clean:
                species_clean, species_group = ("Unknown", "Other")

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

            "species_clean": species_clean,
            "species_group": species_group,

            "moon_phase": moon_phase,
            "moon_illumination": moon_illum,
            "moon_age_days": moon_age,

            "top1_species": top1_label,
            "top1_conf": f"{top1_score:.3f}" if top1_label else "",
            "top2_species": top2_label,
            "top2_conf": f"{top2_score:.3f}" if top2_label else "",
            "top3_species": top3_label,
            "top3_conf": f"{top3_score:.3f}" if top3_label else "",
        }

        was_existing = key in existing
        existing[key] = row
        updated += int(was_existing)
        added += int(not was_existing)

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
