import os
import re
import json
import time
import subprocess
from pathlib import Path

import requests
import spypoint

# Map SpyPoint camera IDs -> your friendly folder names
CAMERA_NAME_MAP = {
    "64d2308abdc2af72ebb0e44b": "gate",
    "696d50430049dd29d16b3c5f": "feeder",
    "696e3766f762a13e8e9526ab": "ravine",
}

# --- Tuning (tight scope) ---
POLL_LIMIT = 120          # how far back to look PER CAMERA
MAX_NEW_PER_RUN = 90      # max NEW images we will download+upload per run (across cameras)
SLEEP_SEC = 0.2           # gentle throttle between downloads

# Local temp download staging (runner is ephemeral anyway)
OUT_DIR = Path("images")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", (s or "").strip()) or "unknown"


def get_cam_id(cam) -> str:
    return str(
        getattr(cam, "id", None)
        or getattr(cam, "camera_id", None)
        or str(cam)
    )


def cam_folder_name(cam) -> str:
    cam_id = get_cam_id(cam)
    return CAMERA_NAME_MAP.get(cam_id, cam_id)


def spypoint_photo_filename(photo) -> str:
    """
    Derive a stable filename from the photo URL.
    """
    url = photo.url()
    base = url.split("?")[0].split("/")[-1]
    return safe_name(base)


def run_rclone_json(args: list[str]) -> list[dict]:
    """
    Run `rclone ... --json` and return parsed JSON list.
    """
    p = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        # Often means folder doesn't exist yet. Treat as empty.
        return []
    out = (p.stdout or "").strip()
    if not out:
        return []
    try:
        return json.loads(out)
    except Exception:
        return []


def drive_existing_filenames(root_folder_id: str, cam_folder: str) -> set[str]:
    """
    List filenames that already exist in Drive under <root>/<cam_folder>.
    Uses rclone lsjson for a fast directory listing.
    """
    # Ensure folder exists (safe to call even if it exists)
    subprocess.run(
        ["rclone", "mkdir", f"gdrive:{cam_folder}", "--drive-root-folder-id", root_folder_id],
        check=False,
    )

    items = run_rclone_json([
        "rclone",
        "lsjson",
        f"gdrive:{cam_folder}",
        "--drive-root-folder-id",
        root_folder_id,
    ])

    existing = set()
    for it in items:
        # rclone returns {"Name": "...", "Size": ..., "IsDir": false, ...}
        name = it.get("Name")
        is_dir = it.get("IsDir", False)
        if name and not is_dir:
            existing.add(str(name))
    return existing


def download(url: str, out_path: Path) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)


def upload_to_drive(local_path: Path, root_folder_id: str, cam_folder: str, filename: str) -> None:
    """
    Upload to: Drive root/<cam_folder>/<filename>
    """
    dest = f"gdrive:{cam_folder}/{filename}"
    subprocess.run(
        [
            "rclone",
            "copyto",
            str(local_path),
            dest,
            "--drive-root-folder-id",
            root_folder_id,
            "-v",
        ],
        check=True,
    )


def main():
    email = os.environ.get("SPYPOINT_EMAIL")
    password = os.environ.get("SPYPOINT_PASSWORD")
    if not email or not password:
        raise SystemExit("Set SPYPOINT_EMAIL and SPYPOINT_PASSWORD env vars.")

    # This must be the Drive folder ID you already use as your "root"
    root_folder_id = os.environ.get("GDRIVE_FOLDER_ID")
    if not root_folder_id:
        raise SystemExit("Set GDRIVE_FOLDER_ID env var (Drive folder that holds your camera folders).")

    c = spypoint.Client(email, password)
    cams = c.cameras()

    print("Camera count:", len(cams))
    for cam in cams:
        cam_id = get_cam_id(cam)
        print("cam:", cam_folder_name(cam), "id:", cam_id)

    # Build a Drive index of existing filenames per camera folder
    existing_by_cam: dict[str, set[str]] = {}
    for cam in cams:
        folder = safe_name(cam_folder_name(cam))
        existing_by_cam[folder] = drive_existing_filenames(root_folder_id, folder)
        print(f"[Drive] {folder}: {len(existing_by_cam[folder])} files indexed")

    new_uploaded = 0
    inspected = 0
    skipped_existing = 0

    for cam in cams:
        folder = safe_name(cam_folder_name(cam))

        # Pull recent photos for this camera
        photos = c.photos([cam], limit=POLL_LIMIT)

        for p in photos:
            inspected += 1
            url = p.url()
            filename = spypoint_photo_filename(p)

            # If it already exists in Drive, skip (does NOT count against MAX_NEW_PER_RUN)
            if filename in existing_by_cam[folder]:
                skipped_existing += 1
                continue

            # Download locally
            cam_dir = OUT_DIR / folder
            cam_dir.mkdir(parents=True, exist_ok=True)
            out_path = cam_dir / filename

            # Even though runner is fresh, this avoids duplicate within-run downloads
            if out_path.exists():
                continue

            try:
                download(url, out_path)
                upload_to_drive(out_path, root_folder_id, folder, filename)
            except Exception as e:
                print(f"[ERROR] {folder}/{filename}: {e}")
                # keep going; transient errors happen
                continue

            # Mark as existing immediately so we don't re-upload within the same run
            existing_by_cam[folder].add(filename)

            new_uploaded += 1
            print(f"[NEW] Uploaded: {folder}/{filename} (new_uploaded={new_uploaded})")

            if new_uploaded >= MAX_NEW_PER_RUN:
                print(f"Reached MAX_NEW_PER_RUN={MAX_NEW_PER_RUN}. Stopping.")
                print(f"Inspected={inspected}  SkippedExisting={skipped_existing}  NewUploaded={new_uploaded}")
                return

            time.sleep(SLEEP_SEC)

    print("Done.")
    print(f"Inspected={inspected}  SkippedExisting={skipped_existing}  NewUploaded={new_uploaded}")


if __name__ == "__main__":
    main()
