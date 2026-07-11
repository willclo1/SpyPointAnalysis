from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from config import PipelineConfig, load_config


def _run_rclone_lsjson(camera: str, root_folder_id: str) -> List[dict]:
    command = [
        "rclone", "lsjson", f"gdrive:{camera}",
        "--drive-root-folder-id", root_folder_id,
        "--files-only",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"rclone failed for {camera}: {(result.stderr or '').strip()}")
    try:
        value = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"rclone returned invalid JSON for {camera}") from exc
    return value if isinstance(value, list) else []


def build_drive_index(cameras: Iterable[str], root_folder_id: str, workers_requested: int) -> Dict[str, Dict[str, str]]:
    camera_list = sorted(set(cameras))
    index: Dict[str, Dict[str, str]] = {}
    workers = max(1, min(workers_requested, len(camera_list) or 1))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_rclone_lsjson, camera, root_folder_id): camera for camera in camera_list}
        for future in as_completed(futures):
            camera = futures[future]
            items = future.result()
            index[camera] = {
                str(item["Name"]): str(item.get("ID") or item.get("Id"))
                for item in items
                if item.get("Name") and (item.get("ID") or item.get("Id"))
            }
            print(f"[DriveIndex] {camera}: {len(index[camera])} files")
    return index


def _load_events(events_csv: Path) -> pd.DataFrame:
    if not events_csv.exists():
        raise SystemExit(f"Missing {events_csv}")

    frame = pd.read_csv(events_csv, dtype=str, keep_default_na=False)
    required = {"camera", "filename", "date", "time", "event_type", "species_clean"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"events.csv is missing required columns: {', '.join(missing)}")

    frame["datetime"] = pd.to_datetime(
        frame["date"].str.strip() + " " + frame["time"].str.strip(),
        format="%m/%d/%Y %I:%M %p",
        errors="coerce",
    )
    frame["event_type"] = frame["event_type"].str.strip().str.lower()
    frame["camera"] = frame["camera"].str.strip()
    frame["filename"] = frame["filename"].str.strip()
    frame["species_clean"] = frame["species_clean"].str.strip()
    frame["species_conf_num"] = pd.to_numeric(frame.get("species_conf", "0"), errors="coerce").fillna(0.0)

    frame = frame[
        (frame["event_type"] == "animal")
        & frame["datetime"].notna()
        & frame["camera"].ne("")
        & frame["filename"].ne("")
        & frame["species_clean"].ne("")
        & frame["species_clean"].str.lower().ne("other")
    ].copy()

    # Keep one row per physical image, preferring the row with the strongest species score.
    frame = frame.sort_values("species_conf_num", ascending=False).drop_duplicates(["camera", "filename"])
    return frame.sort_values(["camera", "datetime", "filename"]).reset_index(drop=True)


def _dominant_species(group: pd.DataFrame) -> str:
    scores: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in group.itertuples(index=False):
        species = str(row.species_clean)
        confidence = max(float(row.species_conf_num), 0.05)
        scores[species] = scores.get(species, 0.0) + confidence
        counts[species] = counts.get(species, 0) + 1
    return max(scores, key=lambda species: (scores[species], counts[species], species))


def _stable_event_id(camera: str, start: pd.Timestamp, filenames: Iterable[str]) -> str:
    payload = f"{camera}|{start.isoformat()}|{'|'.join(filenames)}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def group_into_events(frame: pd.DataFrame, config: PipelineConfig) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    for camera, camera_rows in frame.groupby("camera", sort=False):
        camera_rows = camera_rows.sort_values("datetime").reset_index(drop=True)
        event_gap_minutes = config.camera(str(camera)).event_gap_minutes
        split_marker = camera_rows["datetime"].diff().dt.total_seconds().div(60).gt(event_gap_minutes).cumsum()

        for _, burst in camera_rows.groupby(split_marker, sort=False):
            if len(burst) < config.min_items_per_event:
                continue

            burst = burst.sort_values("datetime")
            species = _dominant_species(burst)
            start = burst["datetime"].iloc[0]
            end = burst["datetime"].iloc[-1]
            all_filenames = burst["filename"].astype(str).tolist()
            shown = burst.head(config.max_items_per_event)

            items = [
                {
                    "datetime": row.datetime.isoformat(),
                    "filename": str(row.filename),
                    "file_id": str(row.file_id),
                    "species": str(row.species_clean),
                    "species_conf": round(float(row.species_conf_num), 3),
                }
                for row in shown.itertuples(index=False)
            ]
            thumbnail_id = next((item["file_id"] for item in items if item["file_id"]), "")

            events.append({
                "event_id": _stable_event_id(str(camera), start, all_filenames),
                "camera": str(camera),
                "species": species,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "duration_minutes": round((end - start).total_seconds() / 60, 1),
                "count": len(burst),
                "items_truncated": len(burst) > config.max_items_per_event,
                "event_gap_minutes": event_gap_minutes,
                "thumbnail_file_id": thumbnail_id,
                "items": items,
            })

    return sorted(events, key=lambda event: event["start"], reverse=True)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    config = load_config()
    root_folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip()
    if not root_folder_id:
        raise SystemExit("Set GDRIVE_FOLDER_ID")

    frame = _load_events(config.events_csv)
    drive_index = build_drive_index(frame["camera"].tolist(), root_folder_id, config.rclone_workers)
    frame["file_id"] = [drive_index.get(camera, {}).get(filename, "") for camera, filename in zip(frame["camera"], frame["filename"])]

    events = group_into_events(frame, config)
    _atomic_write_json(config.events_json, {
        "schema_version": 3,
        "pipeline_version": config.pipeline_version,
        "default_event_gap_minutes": config.default_camera.event_gap_minutes,
        "events": events,
    })

    missing_ids = sum(not item["file_id"] for event in events for item in event["items"])
    print(f"[OK] Wrote {config.events_json}: {len(events)} events")
    print(f"[OK] Missing Drive IDs: {missing_ids}")


if __name__ == "__main__":
    main()
