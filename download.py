import os
import re
import json
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, date, timezone

import requests
import spypoint

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None


# Map SpyPoint camera IDs -> your friendly folder names
CAMERA_NAME_MAP = {
    "696e3766f762a13e8e9526ab": "gate",
    "696d50430049dd29d16b3c5f": "feeder",
    "64d2308abdc2af72ebb0e44b": "creek",
}

# --- Tuning ---
POLL_LIMIT = 250          # lookback window from SpyPoint API (we hard-filter by dates anyway)
MAX_NEW_PER_RUN = 400     # cap across all cameras
SLEEP_SEC = 0.2

# Full mode (within date window): re-download + re-upload even if already in Drive
FULL_REDOWNLOAD = os.environ.get("FULL_REDOWNLOAD") == "1"

# Safety: if a photo has NO parseable date, skip it to prevent accidental backfill
SKIP_IF_NO_DATE = True

# Ranch timezone for date filtering
LOCAL_TZ_NAME = os.environ.get("LOCAL_TZ", "America/Chicago")
LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME) if ZoneInfo else None

# Optional date range
# If neither is set => today-only
START_DATE_ENV = os.environ.get("START_DATE", "").strip()
END_DATE_ENV = os.environ.get("END_DATE", "").strip()

OUT_DIR = Path("images")


# -----------------------------
# Helpers
# -----------------------------
_WS_RE = re.compile(r"\s+")
_DATE_IN_TEXT = re.compile(r"(20\d{2})[-_/]?([01]\d)[-_/]?([0-3]\d)")


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", (s or "").strip()) or "unknown"


def get_cam_id(cam) -> str:
    return str(getattr(cam, "id", None) or getattr(cam, "camera_id", None) or str(cam))


def cam_folder_name(cam) -> str:
    cam_id = get_cam_id(cam)
    return CAMERA_NAME_MAP.get(cam_id, cam_id)


def spypoint_photo_filename(photo) -> str:
    url = photo.url()
    base = url.split("?")[0].split("/")[-1]
    return safe_name(base)


def _now_local_date() -> date:
    if LOCAL_TZ:
        return datetime.now(LOCAL_TZ).date()
    return datetime.now(timezone.utc).date()


def _to_local_date(dt: datetime) -> date:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if LOCAL_TZ:
        return dt.astimezone(LOCAL_TZ).date()
    return dt.astimezone(timezone.utc).date()


def _try_parse_datetime(val) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    s = str(val).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def photo_datetime(photo) -> datetime | None:
    # 1) Common attribute names
    for attr in ("datetime", "date", "created_at", "createdAt", "taken_at", "takenAt", "timestamp"):
        if hasattr(photo, attr):
            dt = _try_parse_datetime(getattr(photo, attr))
            if dt:
                return dt

    # 2) dict-like
    if isinstance(photo, dict):
        for k in ("datetime", "date", "created_at", "createdAt", "taken_at", "takenAt", "timestamp"):
            dt = _try_parse_datetime(photo.get(k))
            if dt:
                return dt

    # 3) Parse date from URL / filename
    try:
        url = photo.url()
    except Exception:
        url = ""
    text = f"{url} {spypoint_photo_filename(photo)}"
    m = _DATE_IN_TEXT.search(text)
    if m:
        y, mo, d = m.groups()
        try:
            if LOCAL_TZ:
                return datetime(int(y), int(mo), int(d), 0, 0, tzinfo=LOCAL_TZ)
            return datetime(int(y), int(mo), int(d), 0, 0, tzinfo=timezone.utc)
        except Exception:
            return None

    return None


def parse_date_env(s: str) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        raise SystemExit(f"Invalid date '{s}'. Use YYYY-MM-DD (e.g. 2026-01-23).")


def resolve_date_window() -> tuple[date, date]:
    """
    Safety-first policy:
    - If neither START_DATE nor END_DATE set: window is [today, today]
    - If START_DATE set but END_DATE not: window is [START_DATE, today]
    - If END_DATE set but START_DATE not: window is [END_DATE, END_DATE]
    - If both set: [START_DATE, END_DATE]
    """
    today = _now_local_date()
    start = parse_date_env(START_DATE_ENV)
    end = parse_date_env(END_DATE_ENV)

    if start is None and end is None:
        return today, today

    if start is not None and end is None:
        return start, today

    if start is None and end is not None:
        return end, end

    # both set
    assert start is not None and end is not None
    if start > end:
        raise SystemExit(f"START_DATE ({start}) cannot be after END_DATE ({end}).")
    return start, end


def in_date_window(photo) -> bool:
    dt = photo_datetime(photo)
    if not dt:
        return False if SKIP_IF_NO_DATE else True
    d = _to_local_date(dt)
    return WINDOW_START <= d <= WINDOW_END


# -----------------------------
# rclone helpers
# -----------------------------
def run_rclone_json(args: list[str]) -> list[dict]:
    p = subprocess.run(args, check=False, capture_output=True, text=True)
    if p.returncode != 0:
        return []
    out = (p.stdout or "").strip()
    if not out:
        return []
    try:
        return json.loads(out)
    except Exception:
        return []


def drive_existing_filenames(root_folder_id: str, cam_folder: str) -> set[str]:
    subprocess.run(
        ["rclone", "mkdir", f"gdrive:{cam_folder}", "--drive-root-folder-id", root_folder_id],
        check=False,
    )

    items = run_rclone_json(
        ["rclone", "lsjson", f"gdrive:{cam_folder}", "--drive-root-folder-id", root_folder_id]
    )

    existing = set()
    for it in items:
        name = it.get("Name")
        is_dir = it.get("IsDir", False)
        if name and not is_dir:
            existing.add(str(name))
    return existing


# -----------------------------
# IO
# -----------------------------
def download(url: str, out_path: Path) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def upload_to_drive(local_path: Path, root_folder_id: str, cam_folder: str, filename: str) -> None:
    dest = f"gdrive:{cam_folder}/{filename}"
    subprocess.run(
        ["rclone", "copyto", str(local_path), dest, "--drive-root-folder-id", root_folder_id, "-v"],
        check=True,
    )


def prepare_local_dirs():
    if FULL_REDOWNLOAD and OUT_DIR.exists():
        print("[MODE] FULL_REDOWNLOAD=1 -> wiping local images/ staging dir")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Main
# -----------------------------
WINDOW_START, WINDOW_END = resolve_date_window()


def main():
    email = os.environ.get("SPYPOINT_EMAIL")
    password = os.environ.get("SPYPOINT_PASSWORD")
    if not email or not password:
        raise SystemExit("Set SPYPOINT_EMAIL and SPYPOINT_PASSWORD env vars.")

    root_folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    if not root_folder_id:
        raise SystemExit("Set GDRIVE_FOLDER_ID env var (Drive folder that holds your camera folders).")

    prepare_local_dirs()

    print(f"[TZ] {LOCAL_TZ_NAME}")
    print(f"[WINDOW] {WINDOW_START} -> {WINDOW_END}")
    if START_DATE_ENV or END_DATE_ENV:
        print("[MODE] Date range requested via START_DATE/END_DATE")
    else:
        print("[MODE] Default: today-only (no START_DATE/END_DATE set)")

    if FULL_REDOWNLOAD:
        print("[MODE] FULL_REDOWNLOAD=1 (re-upload within date window, ignore Drive existence checks)")

    c = spypoint.Client(email, password)
    cams = c.cameras()

    print("Camera count:", len(cams))
    for cam in cams:
        print("cam:", cam_folder_name(cam), "id:", get_cam_id(cam))

    existing_by_cam: dict[str, set[str]] = {}
    if not FULL_REDOWNLOAD:
        for cam in cams:
            folder = safe_name(cam_folder_name(cam))
            existing_by_cam[folder] = drive_existing_filenames(root_folder_id, folder)
            print(f"[Drive] {folder}: {len(existing_by_cam[folder])} files indexed")
    else:
        for cam in cams:
            folder = safe_name(cam_folder_name(cam))
            existing_by_cam[folder] = set()

    new_uploaded = 0
    inspected = 0
    skipped_existing = 0
    skipped_outside_window = 0
    skipped_no_date = 0

    for cam in cams:
        folder = safe_name(cam_folder_name(cam))
        photos = c.photos([cam], limit=POLL_LIMIT)

        for p in photos:
            inspected += 1

            dt = photo_datetime(p)
            if not dt and SKIP_IF_NO_DATE:
                skipped_no_date += 1
                continue

            if not in_date_window(p):
                skipped_outside_window += 1
                continue

            url = p.url()
            filename = spypoint_photo_filename(p)

            if not FULL_REDOWNLOAD and filename in existing_by_cam[folder]:
                skipped_existing += 1
                continue

            cam_dir = OUT_DIR / folder
            cam_dir.mkdir(parents=True, exist_ok=True)
            out_path = cam_dir / filename

            # within-run dedupe
            if out_path.exists():
                continue

            try:
                download(url, out_path)
                upload_to_drive(out_path, root_folder_id, folder, filename)
            except Exception as e:
                print(f"[ERROR] {folder}/{filename}: {e}")
                continue

            existing_by_cam[folder].add(filename)
            new_uploaded += 1

            mode_tag = "REUP" if FULL_REDOWNLOAD else "NEW"
            dt_str = dt.isoformat() if dt else "unknown-dt"
            print(f"[{mode_tag}] {folder}/{filename}  dt={dt_str}  uploaded={new_uploaded}")

            if new_uploaded >= MAX_NEW_PER_RUN:
                print(f"Reached MAX_NEW_PER_RUN={MAX_NEW_PER_RUN}. Stopping.")
                break

            time.sleep(SLEEP_SEC)

        if new_uploaded >= MAX_NEW_PER_RUN:
            break

    print("Done.")
    print(
        f"Inspected={inspected}  "
        f"SkippedOutsideWindow={skipped_outside_window}  "
        f"SkippedNoDate={skipped_no_date}  "
        f"SkippedExisting={skipped_existing}  "
        f"Uploaded={new_uploaded}"
    )


if __name__ == "__main__":
    main()
