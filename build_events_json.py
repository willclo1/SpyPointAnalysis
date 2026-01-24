import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

# -----------------------
# Config
# -----------------------
EVENTS_CSV = Path("events.csv")          # workflow downloads this
OUT_JSON = Path("docs/events.json")      # external site reads this
TIME_GAP_MINUTES = 12                    # split event if gap exceeds this
MAX_ITEMS_PER_EVENT = 80                 # safety cap (optional)

THUMB_SIZE = "w600"  # for drive thumbnails


def thumb_url(file_id: str) -> str:
    return f"https://drive.google.com/thumbnail?id={file_id}&sz={THUMB_SIZE}"


def run_rclone_lsjson(cam_folder: str, root_folder_id: str) -> List[dict]:
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
    Returns: { camera: { filename: file_id } }
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


def load_and_normalize_events() -> pd.DataFrame:
    if not EVENTS_CSV.exists():
        raise SystemExit(f"Missing {EVENTS_CSV}. Did you download it in the workflow?")

    df = pd.read_csv(EVENTS_CSV)

    # Your schema has date + time, e.g. "01/23/2026" + "03:00 AM"
    dt = pd.to_datetime(
        df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
        format="%m/%d/%Y %I:%M %p",
        errors="coerce",
    )
    df = df.copy()
    df["datetime"] = dt

    # Clean text cols
    df["camera"] = df["camera"].astype(str).str.strip()
    df["filename"] = df["filename"].astype(str).str.strip()
    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()

    # This is your clean wildlife label column
    if "species_clean" not in df.columns:
        raise SystemExit("events.csv is missing 'species_clean' column.")

    df["species_clean"] = df["species_clean"].astype(str).str.strip()

    # Drop unusable rows early
    df = df.dropna(subset=["datetime"])
    df = df[(df["camera"] != "") & (df["filename"] != "")]

    # -----------------------
    # YOUR FILTER RULES
    # -----------------------
    # Only wildlife (this automatically discards human/vehicle/blank)
    df = df[df["event_type"] == "animal"].copy()

    # Discard "Other"
    df = df[df["species_clean"].str.lower() != "other"].copy()

    return df


def group_into_events(df: pd.DataFrame, drive_index: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Group by (camera, species_clean) and split into events by time gaps.
    Discard events with size 1.
    """
    events: List[Dict[str, Any]] = []

    df = df.sort_values(["camera", "species_clean", "datetime"]).reset_index(drop=True)

    def get_file_id(cam: str, filename: str) -> str:
        return drive_index.get(cam, {}).get(filename, "")

    for (cam, species), g in df.groupby(["camera", "species_clean"], sort=False):
        g = g.sort_values("datetime").reset_index(drop=True)

        cur_items: List[Dict[str, Any]] = []
        start_dt = None
        last_dt = None

        def flush():
            nonlocal cur_items, start_dt, last_dt
            if not cur_items or start_dt is None or last_dt is None:
                return

            # DROP events of size 1 (your requirement)
            if len(cur_items) <= 1:
                cur_items = []
                start_dt = None
                last_dt = None
                return

            # Pick a thumbnail: first item with a file_id, else blank
            thumb = ""
            for it in cur_items:
                if it["file_id"]:
                    thumb = thumb_url(it["file_id"])
                    break

            event_id = f"{cam}|{species}|{start_dt.isoformat(timespec='minutes')}"
            events.append(
                {
                    "event_id": event_id,
                    "camera": cam,
                    "species": species,
                    "start": start_dt.isoformat(),
                    "end": last_dt.isoformat(),
                    "count": len(cur_items),
                    "thumbnail": thumb,
                    "items": cur_items[:MAX_ITEMS_PER_EVENT],
                }
            )

            cur_items = []
            start_dt = None
            last_dt = None

        for _, r in g.iterrows():
            dt = r["datetime"].to_pydatetime()
            fn = str(r["filename"])

            item = {
                "datetime": dt.isoformat(),
                "filename": fn,
                "file_id": get_file_id(cam, fn),
            }

            if start_dt is None:
                start_dt = dt
                last_dt = dt
                cur_items = [item]
                continue

            gap_min = (dt - last_dt).total_seconds() / 60.0
            if gap_min > TIME_GAP_MINUTES:
                flush()
                start_dt = dt
                last_dt = dt
                cur_items = [item]
            else:
                cur_items.append(item)
                last_dt = dt

        flush()

    # Newest first
    events.sort(key=lambda e: e["start"], reverse=True)
    return events


def main():
    root_folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    if not root_folder_id:
        raise SystemExit("Set GDRIVE_FOLDER_ID env var (Drive folder holding camera folders).")

    df = load_and_normalize_events()
    if df.empty:
        print("[OK] No wildlife (non-Other) rows found; writing empty events.json")
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps({"events": []}, indent=2), encoding="utf-8")
        return

    cameras = sorted(df["camera"].dropna().unique().tolist())
    drive_index = build_drive_index(cameras, root_folder_id)

    events = group_into_events(df, drive_index)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"events": events}, indent=2), encoding="utf-8")

    # Diagnostics
    total_items = sum(e["count"] for e in events)
    missing_ids = sum(1 for e in events for it in e["items"] if not it["file_id"])
    print(f"[OK] Wrote {OUT_JSON} with {len(events)} events, {total_items} items")
    print(f"[Info] Missing Drive file_id for {missing_ids} items (they won't thumbnail)")


if __name__ == "__main__":
    main()
