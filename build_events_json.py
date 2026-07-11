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

EVENTS_CSV = Path(os.environ.get("EVENTS_CSV", "events.csv"))
OUT_JSON = Path(os.environ.get("EVENTS_JSON", "docs/events.json"))
EVENT_GAP_MINUTES = int(os.environ.get("EVENT_GAP_MINUTES", "12"))
MAX_ITEMS_PER_EVENT = int(os.environ.get("MAX_ITEMS_PER_EVENT", "80"))
MIN_ITEMS_PER_EVENT = int(os.environ.get("MIN_ITEMS_PER_EVENT", "2"))
RCLONE_WORKERS = int(os.environ.get("RCLONE_WORKERS", "4"))


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


def build_drive_index(cameras: Iterable[str], root_folder_id: str) -> Dict[str, Dict[str, str]]:
    camera_list = sorted(set(cameras))
    index: Dict[str, Dict[str, str]] = {}
    workers = max(1, min(RCLONE_WORKERS, len(camera_list) or 1))

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


def _load_events() -> pd.DataFrame:
    if not EVENTS_CSV.exists():
        raise SystemExit(f"Missing {EVENTS_CSV}")

    frame = pd.read_csv(EVENTS_CSV, dtype=str, keep_default_na=False)
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


def group_into_events(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    for camera, camera_rows in frame.groupby("camera", sort=False):
        camera_rows = camera_rows.sort_values("datetime").reset_index(drop=True)
        split_marker = camera_rows["datetime"].diff().dt.total_seconds().div(60).gt(EVENT_GAP_MINUTES).cumsum()

        for _, burst in camera_rows.groupby(split_marker, sort=False):
            if len(burst) < MIN_ITEMS_PER_EVENT:
                continue

            burst = burst.sort_values("datetime")
            species = _dominant_species(burst)
            start = burst["datetime"].iloc[0]
            end = burst["datetime"].iloc[-1]
            all_filenames = burst["filename"].astype(str).tolist()
            shown = burst.head(MAX_ITEMS_PER_EVENT)

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
                "items_truncated": len(burst) > MAX_ITEMS_PER_EVENT,
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
    root_folder_id = os.environ.get("GDRIVE_FOLDER_ID", "").strip()
    if not root_folder_id:
        raise SystemExit("Set GDRIVE_FOLDER_ID")

    frame = _load_events()
    drive_index = build_drive_index(frame["camera"].tolist(), root_folder_id)
    frame["file_id"] = [drive_index.get(camera, {}).get(filename, "") for camera, filename in zip(frame["camera"], frame["filename"])]

    events = group_into_events(frame)
    _atomic_write_json(OUT_JSON, {
        "schema_version": 2,
        "event_gap_minutes": EVENT_GAP_MINUTES,
        "events": events,
    })

    missing_ids = sum(not item["file_id"] for event in events for item in event["items"])
    print(f"[OK] Wrote {OUT_JSON}: {len(events)} events")
    print(f"[OK] Missing Drive IDs: {missing_ids}")


if __name__ == "__main__":
    main()
