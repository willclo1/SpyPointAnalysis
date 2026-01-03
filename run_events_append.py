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

# ✅ Add taxonomy file (commit this into your repo)
# Download once:
#   curl -L -o speciesnet_taxonomy.json https://raw.githubusercontent.com/google/cameratrapai/main/speciesnet/data/taxonomy.json
TAXONOMY_JSON = Path("speciesnet_taxonomy.json")

UPDATE_EXISTING = os.environ.get("UPDATE_EXISTING") == "1"

# thresholds (detections come from SpeciesNet output)
ANIMAL_THRESH = 0.20
HUMAN_THRESH = 0.30
VEHICLE_THRESH = 0.30

CAT_ANIMAL = "1"
CAT_HUMAN = "2"
CAT_VEHICLE = "3"


FIELDS = [
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


def load_existing(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Loads existing rows keyed by filename. Handles both old and new schemas gracefully.
    """
    if not csv_path.exists():
        return {}

    rows: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fn = r.get("filename")
            if not fn:
                continue
            normalized = {k: (r.get(k, "") or "") for k in FIELDS}
            rows[fn] = normalized
    return rows


def write_table(path: Path, rows: List[Dict[str, str]], delimiter: str):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=FIELDS,
            delimiter=delimiter,
            quoting=csv.QUOTE_MINIMAL
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


# ✅ Taxonomy loading + mapping (IDs -> human-readable labels)
def load_taxonomy_map(taxonomy_path: Path) -> Dict[str, str]:
    """
    Builds an id -> label mapping from SpeciesNet taxonomy.json.

    Preference order:
      common_name > scientific_name > name > id
    """
    if not taxonomy_path.exists():
        # If taxonomy isn't present, we still run, but you'll see IDs.
        return {}

    with taxonomy_path.open("r") as f:
        data = json.load(f)

    mapping: Dict[str, str] = {}

    # taxonomy.json from cameratrapai/speciesnet uses a "nodes" array
    for node in data.get("nodes", []) or []:
        node_id = node.get("id")
        if not node_id:
            continue

        label = (
            node.get("common_name")
            or node.get("scientific_name")
            or node.get("name")
            or node_id
        )
        mapping[str(node_id)] = str(label)

    return mapping


def map_label(raw: str, taxonomy: Dict[str, str]) -> str:
    if not raw:
        return ""
    return taxonomy.get(raw, raw)


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


def extract_top3(pred: dict, taxonomy: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    Returns up to 3 (label, conf_string) tuples from SpeciesNet classifications, if present.
    Converts taxonomy IDs to readable names if taxonomy is available.
    """
    cls = pred.get("classifications", {}) or {}
    classes = cls.get("classes", []) or []
    scores = cls.get("scores", []) or []

    out: List[Tuple[str, str]] = []
    for c, s in list(zip(classes, scores))[:3]:
        label = map_label(str(c), taxonomy)
        try:
            out.append((label, f"{float(s):.3f}"))
        except Exception:
            out.append((label, str(s)))
    return out


def main():
    if not IMAGES_DIR.exists():
        raise SystemExit("Missing images folder")

    # ✅ Load taxonomy mapping (if file is present)
    taxonomy = load_taxonomy_map(TAXONOMY_JSON)

    # Run SpeciesNet once for the whole folder
    run_speciesnet(IMAGES_DIR, SPECIESNET_JSON)

    with SPECIESNET_JSON.open("r") as f:
        sn = json.load(f)

    preds = sn.get("predictions", [])
    by_filename: Dict[str, dict] = {}
    for p in preds:
        fp = p.get("filepath", "")
        if fp:
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

        animal_conf = max_conf_for_category(dets, CAT_ANIMAL)
        human_conf = max_conf_for_category(dets, CAT_HUMAN)
        vehicle_conf = max_conf_for_category(dets, CAT_VEHICLE)

        event_type = pick_event_type(animal_conf, human_conf, vehicle_conf)

        # SpeciesNet final ensemble prediction + score (meaningful for animals)
        raw_species = pred.get("prediction", "") or ""
        species = map_label(raw_species, taxonomy)

        score_val = pred.get("prediction_score", None)
        species_conf = "" if score_val is None else f"{float(score_val):.3f}"

        # Top 3 (ID -> label)
        top3 = extract_top3(pred, taxonomy)
        while len(top3) < 3:
            top3.append(("", ""))

        # If it’s not an animal event, make the label human-friendly
        if event_type in ("human", "vehicle"):
            species = event_type
            species_conf = f"{(human_conf if event_type == 'human' else vehicle_conf):.3f}"
            top3 = [("", ""), ("", ""), ("", "")]
        elif event_type == "blank":
            species = ""
            species_conf = ""
            top3 = [("", ""), ("", ""), ("", "")]
        else:
            # animal event but prediction empty: fallback to top1 if available
            if not species and top3[0][0]:
                species = top3[0][0]
                species_conf = top3[0][1]

        row = {
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

        was_existing = fn in existing
        existing[fn] = row
        if was_existing:
            updated += 1
        else:
            added += 1

    all_rows = [existing[k] for k in sorted(existing.keys())]

    # Write both formats
    write_table(OUT_CSV, all_rows, delimiter=",")
    write_table(OUT_TSV, all_rows, delimiter="\t")

    print(f"Wrote {OUT_CSV} and {OUT_TSV} with {len(all_rows)} rows")
    print(f"Added {added}, updated {updated}")


if __name__ == "__main__":
    main()
