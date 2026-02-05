import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

# -----------------------
# Config
# -----------------------
EVENTS_CSV = Path("events.csv")          # your workflow downloads this
OUT_JSON = Path("docs/events.json")      # output for your site
EVENT_GAP_MINUTES = 12                   # split events if gap exceeds this
MAX_ITEMS_PER_EVENT = 80                 # safety cap for huge bursts


def run_rclone_lsjson(cam_folder: str, root_folder_id: str) -> List[dict]:
    """
    Lists files in gdrive:<cam_folder> and returns rclone lsjson output.
    For Google Drive, rclone includes "ID" for files.
    """
    cmd = [
        "rclone",
        "lsjson",
        f"gdrive:{cam_folder}",
        "--drive-root-folder-id",
        root_folder_id,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return []
    out = (p.stdout or "").strip()
    if not out:
        return []
    try:
        return json.loads(out)
    except Exception:
        return []


def build_drive_index(cameras: List[str], root_folder_id: str) -> Dict[str, Dict[str, str]]:
    """
    Returns:
      { camera: { filename: file_id } }
    """
    index: Dict[str, Dict[str, str]] = {}
    for cam in cameras:
        items = run_rclone_lsjson(cam, root_folder_id)
        m: Dict[str, str] = {}
        for it in items:
            if it.get("IsDir"):
                continue
            name = it.get("Name")
            fid = it.get("ID") or it.get("Id")
            if name and fid:
                m[str(name)] = str(fid)
        index[cam] = m
        print(f"[DriveIndex] {cam}: {len(m)} files")
    return index


def parse_datetime_from_date_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Your CSV uses:
      date: "01/23/2026"
      time: "03:00 AM"
    """
    df = df.copy()
    dt = pd.to_datetime(
        df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
        format="%m/%d/%Y %I:%M %p",
        errors="coerce",
    )
    df["datetime"] = dt
    return df


def filter_to_curated_wildlife(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply your rules:
    - discard human / vehicle / blank
    - discard species_clean == Other
    """
    df = df.copy()

    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    df = df[df["event_type"] == "animal"]

    # Species column in your file is species_clean (use that for UI)
    if "species_clean" not in df.columns:
        raise ValueError("Expected column 'species_clean' in events.csv")

    df["species_clean"] = df["species_clean"].astype(str).str.strip()
    df = df[df["species_clean"].notna()]
    df = df[df["species_clean"] != ""]
    df = df[df["species_clean"].str.lower() != "other"]

    # Require key fields
    df["camera"] = df["camera"].astype(str).str.strip()
    df["filename"] = df["filename"].astype(str).str.strip()
    df = df[(df["camera"] != "") & (df["filename"] != "")]

    # Require datetime
    df = df.dropna(subset=["datetime"])

    return df


def group_into_events(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Groups by (camera, species_clean) and splits into events by time gaps.
    Then removes events of size 1 (your requirement).
    """
    df = df.sort_values(["camera", "species_clean", "datetime"]).reset_index(drop=True)

    events: List[Dict[str, Any]] = []

    for (cam, species), g in df.groupby(["camera", "species_clean"], sort=False):
        g = g.sort_values("datetime").reset_index(drop=True)

        current: List[Tuple[pd.Timestamp, str, str]] = []  # (dt, filename, file_id)
        start_dt: Optional[pd.Timestamp] = None
        last_dt: Optional[pd.Timestamp] = None

        def flush():
            nonlocal current, start_dt, last_dt
            if not current or start_dt is None or last_dt is None:
                return

            # DROP size 1 events (your requirement)
            if len(current) <= 1:
                current = []
                start_dt = None
                last_dt = None
                return

            # cap items if huge burst
            items = current[:MAX_ITEMS_PER_EVENT]

            # pick thumb as first item with file_id (if any)
            thumb_id = ""
            for _, _, fid in items:
                if fid:
                    thumb_id = fid
                    break

            event_id = f"{cam}|{species}|{start_dt.isoformat(timespec='minutes')}"

            events.append(
                {
                    "event_id": event_id,
                    "camera": cam,
                    "species": species,
                    "start": start_dt.isoformat(),
                    "end": last_dt.isoformat(),
                    "count": len(current),
                    "thumbnail_file_id": thumb_id,  # your site can build thumb URL from this
                    "items": [
                        {
                            "datetime": dt.isoformat(),
                            "filename": fn,
                            "file_id": fid,
                        }
                        for dt, fn, fid in items
                    ],
                }
            )

            current = []
            start_dt = None
            last_dt = None

        for _, r in g.iterrows():
            dt: pd.Timestamp = r["datetime"]
            fn: str = str(r["filename"])
            fid: str = str(r.get("file_id") or "")

            if start_dt is None:
                start_dt = dt
                last_dt = dt
                current = [(dt, fn, fid)]
                continue

            gap_min = (dt - last_dt).total_seconds() / 60.0 if last_dt is not None else 0.0
            if gap_min > EVENT_GAP_MINUTES:
                flush()
                start_dt = dt
                last_dt = dt
                current = [(dt, fn, fid)]
            else:
                current.append((dt, fn, fid))
                last_dt = dt

        flush()

    # newest first
    events.sort(key=lambda e: e["start"], reverse=True)
    return events


def main():
    if not EVENTS_CSV.exists():
        raise SystemExit(f"Missing {EVENTS_CSV}. (Your workflow should download it from Drive first.)")

    root_folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    if not root_folder_id:
        raise SystemExit("Set GDRIVE_FOLDER_ID env var (Drive folder that contains your camera folders).")

    df = pd.read_csv(EVENTS_CSV)

    # Build datetime from your real columns
    if "date" not in df.columns or "time" not in df.columns:
        raise ValueError(f"events.csv must have 'date' and 'time' columns. Found: {list(df.columns)}")
    df = parse_datetime_from_date_time(df)

    # Curate: wildlife only, no Other
    df = filter_to_curated_wildlife(df)

    # Build Drive index so we can attach file IDs for thumbnails
    cameras = sorted(df["camera"].dropna().unique().tolist())
    drive_index = build_drive_index(cameras, root_folder_id)

    # Attach file_id to each row
    df["file_id"] = df.apply(
        lambda r: drive_index.get(r["camera"], {}).get(r["filename"], ""),
        axis=1,
    )

    # Group into events; drops size-1 events
    events = group_into_events(df)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"events": events}, indent=2), encoding="utf-8")

    # quick summary
    total_items = sum(e["count"] for e in events)
    missing_ids = sum(1 for e in events for it in e["items"] if not it.get("file_id"))
    print(f"[OK] Wrote {OUT_JSON} with {len(events)} events, {total_items} items")
    print(f"[OK] Missing Drive IDs for {missing_ids} items (those will not have thumbnails)")


if __name__ == "__main__":
    main()
