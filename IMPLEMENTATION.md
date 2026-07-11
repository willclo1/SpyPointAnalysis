# Resumable pipeline and camera configuration

## New files

- `config.py`: validated, centralized configuration with environment-variable overrides.
- `pipeline_config.json`: repository-level defaults and optional per-camera overrides.
- `processing_manifest.py`: atomic JSON manifest containing image hashes, status, errors, and cached CSV rows.
- `test_config_manifest.py`: tests for camera inheritance and cache invalidation.

## Existing outputs remain compatible

The Streamlit application can continue reading:

- `events.csv`
- `events.tsv`
- `docs/events.json`

The new `processing-manifest.json` is pipeline metadata and does not replace any Streamlit data source.

## GitHub Actions persistence

The manifest must be downloaded before processing and uploaded afterward, just like `events.csv`.

Add this to the existing download step:

```bash
rclone copyto "gdrive:processing-manifest.json" "processing-manifest.json" \
  --drive-root-folder-id "$GDRIVE_FOLDER_ID" -v || true
```

Add this to the upload step:

```bash
rclone copyto processing-manifest.json "gdrive:processing-manifest.json" \
  --drive-root-folder-id "$GDRIVE_FOLDER_ID" -v
```

For a full rebuild, the script ignores cached rows and recreates the manifest. Increment `pipeline_version` in `pipeline_config.json` whenever a logic change should force all images through processing again.

## Camera names

Camera keys must match the immediate folder name under `images/` and the corresponding Google Drive folder. For example:

```text
images/
  Front Gate/
  North Pasture/
```

would use:

```json
"cameras": {
  "Front Gate": {
    "event_gap_minutes": 8,
    "vehicle_threshold": 0.25
  },
  "North Pasture": {
    "event_gap_minutes": 15,
    "species_fallback_threshold": 0.45
  }
}
```

Unknown cameras automatically use `default_camera`.

## Cache behavior

An image is reused only when all of these are true:

1. Its SHA-256 hash is unchanged.
2. Its manifest status is `complete`.
3. Its manifest pipeline version matches the current configuration.
4. `FULL_REBUILD` and `UPDATE_EXISTING` are disabled.

Legacy CSV rows are automatically imported into the manifest during the first run, avoiding an unnecessary OCR bill.
