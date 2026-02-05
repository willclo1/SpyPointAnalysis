import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from species_normalization import normalize_species, clean_label, is_junk_or_broad
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

# species selection thresholds
SPECIES_PRIMARY_MIN = 0.60    # accept main prediction if it's specific enough
SPECIES_FALLBACK_MIN = 0.25   # accept top candidates if main is junk/broad

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
    "species_conf",
    "species_clean",
    "species_group",

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
    candidates: List[Tuple[str, float]] = []
    if animal_c >= ANIMAL_THRESH:
        candidates.append(("animal", animal_c))
    if human_c >= HUMAN_THRESH:
        candidates.append(("human", human_c))
    if vehicle_c >= VEHICLE_THRESH:
        candidates.append(("vehicle", vehicle_c))

    if not candidates:
        return "blank"

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def extract_top3(pred: dict) -> List[Tuple[str, float]]:
    cls = pred.get("classifications", {}) or {}
    classes = cls.get("classes", []) or []
    scores = cls.get("scores", []) or []

    out: List[Tuple[str, float]] = []
    for c, s in list(zip(classes, scores))[:3]:
        label = last_after_semicolon(c)
        try:
            out.append((label, float(s)))
        except Exception:
            out.append((label, 0.0))
    while len(out) < 3:
        out.append(("", 0.0))
    return out


def choose_best_species(
    species_raw: str,
    species_conf: Optional[float],
    top3: List[Tuple[str, float]],
) -> Tuple[str, float]:
    """
    Decide which label we should treat as the "species" for normalization.

    Rule:
      - If main prediction is specific (not junk/broad) AND conf >= 0.60, use it.
      - Otherwise, look through top1/top2/top3 for the best specific label
        with conf >= 0.25 (fallback threshold).
      - Otherwise return empty ("").
    """
    s0 = clean_label(species_raw)
    c0 = float(species_conf) if species_conf is not None else 0.0

    if s0 and (not is_junk_or_broad(s0)) and c0 >= SPECIES_PRIMARY_MIN:
        return (species_raw, c0)

    # fallback: best specific candidate among top3
    best_label = ""
    best_conf = 0.0
    for lab, conf in top3:
        cl = clean_label(lab)
        if not cl:
            continue
        if is_junk_or_broad(cl):
            continue
        if conf >= SPECIES_FALLBACK_MIN and conf > best_conf:
            best_label, best_conf = lab, conf

    if best_label:
        return (best_label, best_conf)

    return ("", 0.0)


def main():
    if not IMAGES_DIR.exists():
        raise SystemExit("Missing images folder")

    run_speciesnet(IMAGES_DIR, SPECIESNET_JSON)

    with SPECIESNET_JSON.open("r") as f:
        sn = json.load(f)

    preds = sn.get("predictions", []) or []

    # Map predictions by relpath from IMAGES_DIR
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

        rel_key = str(img_path.resolve().relative_to(IMAGES_DIR.resolve()))
        pred = by_relpath.get(rel_key, {})
        dets = pred.get("detections", []) or []

        animal_conf = max_conf_for_category(dets, CAT_ANIMAL)
        human_conf = max_conf_for_category(dets, CAT_HUMAN)
        vehicle_conf = max_conf_for_category(dets, CAT_VEHICLE)

        event_type = pick_event_type(animal_conf, human_conf, vehicle_conf)

        # Defaults
        species = ""
        species_conf_val = ""
        species_clean = "Other"
        species_group = "Other"

        top3 = extract_top3(pred)

        if event_type in ("human", "vehicle"):
            # keep animals-only normalization out of these
            species = event_type
            species_conf_val = f"{(human_conf if event_type == 'human' else vehicle_conf):.3f}"
            species_clean = "Other"
            species_group = "Other"

            top3 = [("", 0.0), ("", 0.0), ("", 0.0)]

        elif event_type == "blank":
            species = ""
            species_conf_val = ""
            species_clean = "Other"
            species_group = "Other"
            top3 = [("", 0.0), ("", 0.0), ("", 0.0)]

        else:
            # animal: pick best label
            raw_species = last_after_semicolon(pred.get("prediction", "") or "")
            score_val = pred.get("prediction_score", None)
            score_float = float(score_val) if score_val is not None else None

            chosen_label, chosen_conf = choose_best_species(raw_species, score_float, top3)

            # Use chosen label if present; else keep raw_species (for debugging) but it may be broad
            species = chosen_label or raw_species or ""
            species_conf_val = f"{(chosen_conf if chosen_label else (score_float or 0.0)):.3f}" if (chosen_label or score_float is not None) else ""

            # Normalize for dashboard
            species_clean, species_group = normalize_species(species)

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
            "species_conf": species_conf_val,
            "species_clean": species_clean,
            "species_group": species_group,

            "top1_species": top3[0][0],
            "top1_conf": f"{top3[0][1]:.3f}",
            "top2_species": top3[1][0],
            "top2_conf": f"{top3[1][1]:.3f}",
            "top3_species": top3[2][0],
            "top3_conf": f"{top3[2][1]:.3f}",
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
