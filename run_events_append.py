from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import CameraConfig, load_config
from moon import moon_info
from processing_manifest import ProcessingManifest
from species_normalization import normalize_species
from vision_ocr import Stamp, ocr_spypoint_stamp_vision

FIELDS = [
    "camera", "filename", "date", "time", "temp_f", "temp_c",
    "event_type", "animal_conf", "human_conf", "vehicle_conf",
    "species", "species_conf", "species_clean", "species_group",
    "moon_phase", "moon_illumination", "moon_age_days",
    "top1_species", "top1_conf", "top2_species", "top2_conf",
    "top3_species", "top3_conf",
]
CAT_ANIMAL = "1"
CAT_HUMAN = "2"
CAT_VEHICLE = "3"


def last_after_semicolon(label: str) -> str:
    return str(label or "").strip().rsplit(";", 1)[-1].strip().replace("_", " ")


def _to_float(value: object) -> float:
    try:
        text = str(value or "").strip()
        return 0.0 if not text or text.lower() in {"nan", "none", "null"} else float(text)
    except (TypeError, ValueError):
        return 0.0


def row_key(camera: str, filename: str) -> str:
    return f"{camera}::{filename}"


def load_existing(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            filename = (row.get("filename") or "").strip()
            camera = (row.get("camera") or "").strip() or "unknown"
            if filename:
                rows[row_key(camera, filename)] = {field: row.get(field, "") or "" for field in FIELDS}
    return rows


def write_table(path: Path, rows: List[Dict[str, str]], delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") or "" for field in FIELDS})
        temporary = Path(handle.name)
    temporary.replace(path)


def run_speciesnet(images_dir: Path, out_json: Path, country: str, admin1: str) -> None:
    command = [
        sys.executable, "-m", "speciesnet.scripts.run_model",
        "--folders", str(images_dir),
        "--predictions_json", str(out_json),
    ]
    if country:
        command += ["--country", country]
    if admin1:
        command += ["--admin1_region", admin1]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"SpeciesNet failed: {(result.stderr or result.stdout).strip()}")


def max_conf_for_category(detections: List[dict], category: str) -> float:
    return max((_to_float(item.get("conf")) for item in detections if item.get("category") == category), default=0.0)


def pick_event_type(animal: float, human: float, vehicle: float, camera: CameraConfig) -> str:
    candidates = []
    if animal >= camera.animal_threshold:
        candidates.append(("animal", animal))
    if human >= camera.human_threshold:
        candidates.append(("human", human))
    if vehicle >= camera.vehicle_threshold:
        candidates.append(("vehicle", vehicle))
    return max(candidates, key=lambda item: item[1])[0] if candidates else "blank"


def extract_top3(prediction: dict) -> List[Tuple[str, float]]:
    classifications = prediction.get("classifications", {}) or {}
    values = [
        (last_after_semicolon(label), _to_float(score))
        for label, score in zip(classifications.get("classes", []) or [], classifications.get("scores", []) or [])
    ][:3]
    return values + [("", 0.0)] * (3 - len(values))


def _is_candidate_usable(label: str) -> bool:
    low = (label or "").strip().lower()
    return bool(low and low not in {"blank", "no cv result", "animal", "bird", "corvus species", "canis species"} and "no cv" not in low)


def choose_best_species_label(prediction: dict, camera: CameraConfig) -> Tuple[str, float]:
    candidates = []
    raw = last_after_semicolon(prediction.get("prediction", ""))
    if raw:
        candidates.append((raw, _to_float(prediction.get("prediction_score"))))
    candidates.extend(extract_top3(prediction))
    usable = [(label, score) for label, score in candidates if _is_candidate_usable(label)]
    strong = [item for item in usable if item[1] >= camera.species_strong_threshold]
    fallback = [item for item in usable if item[1] >= camera.species_fallback_threshold]
    pool = strong or fallback
    return max(pool, key=lambda item: item[1]) if pool else ("", 0.0)


def parse_stamp_datetime(date_value: str, time_value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(f"{date_value.strip()} {time_value.strip()}", "%m/%d/%Y %I:%M %p")
    except (AttributeError, ValueError):
        return None


def moon_fields(stamp: Stamp) -> Tuple[str, str, str]:
    value = parse_stamp_datetime(stamp.date_mmddyyyy or "", stamp.time_hhmm_ampm or "")
    if not value:
        return "", "", ""
    info = moon_info(value)
    return info.phase_name, f"{info.illumination:.3f}", f"{info.age_days:.2f}"


def prediction_index(payload: dict, images_dir: Path) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    root = images_dir.resolve()
    for prediction in payload.get("predictions", []) or []:
        filepath = prediction.get("filepath") or ""
        if not filepath:
            continue
        try:
            key = str(Path(filepath).resolve().relative_to(root))
        except ValueError:
            key = Path(filepath).name
        result[key] = prediction
    return result


def speciesnet_covers_images(index: Dict[str, dict], images: List[Path], root: Path) -> bool:
    expected = {str(path.resolve().relative_to(root.resolve())) for path in images}
    return expected.issubset(index.keys())


def build_row(image: Path, camera_name: str, prediction: dict, camera: CameraConfig) -> Dict[str, str]:
    try:
        stamp = ocr_spypoint_stamp_vision(str(image), crop_fraction=camera.ocr_crop_fraction)
    except RuntimeError as exc:
        print(f"[OCR ERROR] {image}: {exc}")
        stamp = Stamp(None, None, None, None, "")

    phase, illumination, age = moon_fields(stamp)
    detections = prediction.get("detections", []) or []
    animal = max_conf_for_category(detections, CAT_ANIMAL)
    human = max_conf_for_category(detections, CAT_HUMAN)
    vehicle = max_conf_for_category(detections, CAT_VEHICLE)
    event_type = pick_event_type(animal, human, vehicle, camera)
    top3 = extract_top3(prediction)

    if event_type == "human":
        species, score, clean, group = "human", human, "Human", "Human"
    elif event_type == "vehicle":
        species, score, clean, group = "vehicle", vehicle, "Vehicle", "Vehicle"
    elif event_type == "blank":
        species, score, clean, group = "", 0.0, "", ""
    else:
        species, score = choose_best_species_label(prediction, camera)
        clean, group = normalize_species(species)
        if not clean:
            clean, group = "Unknown", "Other"

    return {
        "camera": camera_name,
        "filename": image.name,
        "date": stamp.date_mmddyyyy or "",
        "time": stamp.time_hhmm_ampm or "",
        "temp_f": "" if stamp.temp_f is None else str(stamp.temp_f),
        "temp_c": "" if stamp.temp_c is None else str(stamp.temp_c),
        "event_type": event_type,
        "animal_conf": f"{animal:.3f}",
        "human_conf": f"{human:.3f}",
        "vehicle_conf": f"{vehicle:.3f}",
        "species": species,
        "species_conf": "" if score <= 0 else f"{score:.3f}",
        "species_clean": clean,
        "species_group": group,
        "moon_phase": phase,
        "moon_illumination": illumination,
        "moon_age_days": age,
        "top1_species": top3[0][0], "top1_conf": f"{top3[0][1]:.3f}" if top3[0][0] else "",
        "top2_species": top3[1][0], "top2_conf": f"{top3[1][1]:.3f}" if top3[1][0] else "",
        "top3_species": top3[2][0], "top3_conf": f"{top3[2][1]:.3f}" if top3[2][0] else "",
    }


def main() -> None:
    config = load_config()
    if not config.images_dir.exists():
        raise SystemExit(f"Missing image folder: {config.images_dir}")

    images = sorted(path for path in config.images_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    existing = {} if config.full_rebuild else load_existing(config.events_csv)
    manifest = ProcessingManifest(config.manifest_path, config.pipeline_version)

    species_payload = {}
    index: Dict[str, dict] = {}
    if config.speciesnet_json.exists():
        try:
            species_payload = json.loads(config.speciesnet_json.read_text(encoding="utf-8"))
            index = prediction_index(species_payload, config.images_dir)
        except (OSError, json.JSONDecodeError):
            index = {}

    can_reuse = config.reuse_speciesnet and speciesnet_covers_images(index, images, config.images_dir)
    if not can_reuse:
        run_speciesnet(config.images_dir, config.speciesnet_json, config.speciesnet_country, config.speciesnet_admin1)
        species_payload = json.loads(config.speciesnet_json.read_text(encoding="utf-8"))
        index = prediction_index(species_payload, config.images_dir)
    else:
        print(f"[CACHE] Reusing complete {config.speciesnet_json}")

    counters = {"cached": 0, "processed": 0, "failed": 0}
    valid_keys: set[str] = set()
    for image in images:
        camera_name = image.parent.name if image.parent != config.images_dir else "unknown"
        key = row_key(camera_name, image.name)
        valid_keys.add(key)
        signature = manifest.file_signature(image)

        if not config.full_rebuild and not config.update_existing:
            cached = manifest.cached_row(key, signature)
            if cached:
                existing[key] = {field: cached.get(field, "") or "" for field in FIELDS}
                counters["cached"] += 1
                continue
            if key in existing and not manifest.get(key):
                # Seamlessly import legacy CSV rows into the new manifest.
                manifest.mark_complete(key, signature, existing[key])
                counters["cached"] += 1
                continue

        relative = str(image.resolve().relative_to(config.images_dir.resolve()))
        prediction = index.get(relative) or index.get(image.name) or {}
        try:
            row = build_row(image, camera_name, prediction, config.camera(camera_name))
            existing[key] = row
            manifest.mark_complete(key, signature, row)
            counters["processed"] += 1
        except Exception as exc:
            manifest.mark_failed(key, signature, str(exc))
            counters["failed"] += 1
            print(f"[PROCESS ERROR] {image}: {exc}")

    pruned = manifest.prune(valid_keys) if config.full_rebuild else 0
    rows = [existing[key] for key in sorted(existing, key=lambda value: tuple(value.split("::", 1)))]
    write_table(config.events_csv, rows, ",")
    write_table(config.events_tsv, rows, "\t")
    manifest.save()

    print(f"[OK] Rows: {len(rows)} | processed: {counters['processed']} | cached: {counters['cached']} | failed: {counters['failed']} | pruned: {pruned}")
    print(f"[OK] Manifest: {config.manifest_path}")


if __name__ == "__main__":
    main()
