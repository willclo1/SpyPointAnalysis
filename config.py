from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


CONFIG_PATH = Path(os.environ.get("PIPELINE_CONFIG", "pipeline_config.json"))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


@dataclass(frozen=True)
class CameraConfig:
    event_gap_minutes: int = 12
    animal_threshold: float = 0.20
    human_threshold: float = 0.30
    vehicle_threshold: float = 0.30
    species_strong_threshold: float = 0.60
    species_fallback_threshold: float = 0.35
    ocr_crop_fraction: float = 0.30
    timezone: str = "America/Chicago"


@dataclass(frozen=True)
class PipelineConfig:
    pipeline_version: str = "3.0.0"
    images_dir: Path = Path("images")
    events_csv: Path = Path("events.csv")
    events_tsv: Path = Path("events.tsv")
    events_json: Path = Path("docs/events.json")
    speciesnet_json: Path = Path("speciesnet-results.json")
    manifest_path: Path = Path("processing-manifest.json")
    speciesnet_country: str = "USA"
    speciesnet_admin1: str = "TX"
    reuse_speciesnet: bool = True
    update_existing: bool = False
    full_rebuild: bool = False
    max_items_per_event: int = 80
    min_items_per_event: int = 2
    rclone_workers: int = 4
    default_camera: CameraConfig = field(default_factory=CameraConfig)
    cameras: Dict[str, CameraConfig] = field(default_factory=dict)

    def camera(self, name: str) -> CameraConfig:
        return self.cameras.get(name, self.default_camera)


def _camera_from_dict(value: Dict[str, Any], base: CameraConfig | None = None) -> CameraConfig:
    base = base or CameraConfig()
    return CameraConfig(
        event_gap_minutes=int(value.get("event_gap_minutes", base.event_gap_minutes)),
        animal_threshold=float(value.get("animal_threshold", base.animal_threshold)),
        human_threshold=float(value.get("human_threshold", base.human_threshold)),
        vehicle_threshold=float(value.get("vehicle_threshold", base.vehicle_threshold)),
        species_strong_threshold=float(value.get("species_strong_threshold", base.species_strong_threshold)),
        species_fallback_threshold=float(value.get("species_fallback_threshold", base.species_fallback_threshold)),
        ocr_crop_fraction=float(value.get("ocr_crop_fraction", base.ocr_crop_fraction)),
        timezone=str(value.get("timezone", base.timezone)),
    )


def load_config(path: Path = CONFIG_PATH) -> PipelineConfig:
    raw: Dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a JSON object")
        raw = loaded

    default_camera = _camera_from_dict(raw.get("default_camera", {}) or {})
    cameras = {
        str(name): _camera_from_dict(value or {}, default_camera)
        for name, value in (raw.get("cameras", {}) or {}).items()
    }

    return PipelineConfig(
        pipeline_version=str(raw.get("pipeline_version", "3.0.0")),
        images_dir=Path(os.environ.get("IMAGE_DIRECTORY", raw.get("images_dir", "images"))),
        events_csv=Path(os.environ.get("EVENTS_CSV", raw.get("events_csv", "events.csv"))),
        events_tsv=Path(os.environ.get("EVENTS_TSV", raw.get("events_tsv", "events.tsv"))),
        events_json=Path(os.environ.get("EVENTS_JSON", raw.get("events_json", "docs/events.json"))),
        speciesnet_json=Path(os.environ.get("SPECIESNET_JSON", raw.get("speciesnet_json", "speciesnet-results.json"))),
        manifest_path=Path(os.environ.get("PROCESSING_MANIFEST", raw.get("manifest_path", "processing-manifest.json"))),
        speciesnet_country=os.environ.get("SPECIESNET_COUNTRY", raw.get("speciesnet_country", "USA")),
        speciesnet_admin1=os.environ.get("SPECIESNET_ADMIN1", raw.get("speciesnet_admin1", "TX")),
        reuse_speciesnet=_env_bool("REUSE_SPECIESNET", bool(raw.get("reuse_speciesnet", True))),
        update_existing=_env_bool("UPDATE_EXISTING", bool(raw.get("update_existing", False))),
        full_rebuild=_env_bool("FULL_REBUILD", bool(raw.get("full_rebuild", False))),
        max_items_per_event=_env_int("MAX_ITEMS_PER_EVENT", int(raw.get("max_items_per_event", 80))),
        min_items_per_event=_env_int("MIN_ITEMS_PER_EVENT", int(raw.get("min_items_per_event", 2))),
        rclone_workers=_env_int("RCLONE_WORKERS", int(raw.get("rclone_workers", 4))),
        default_camera=CameraConfig(
            event_gap_minutes=_env_int("EVENT_GAP_MINUTES", default_camera.event_gap_minutes),
            animal_threshold=_env_float("ANIMAL_THRESH", default_camera.animal_threshold),
            human_threshold=_env_float("HUMAN_THRESH", default_camera.human_threshold),
            vehicle_threshold=_env_float("VEHICLE_THRESH", default_camera.vehicle_threshold),
            species_strong_threshold=_env_float("SPECIES_STRONG_THRESH", default_camera.species_strong_threshold),
            species_fallback_threshold=_env_float("SPECIES_FALLBACK_THRESH", default_camera.species_fallback_threshold),
            ocr_crop_fraction=_env_float("OCR_CROP_FRACTION", default_camera.ocr_crop_fraction),
            timezone=os.environ.get("PIPELINE_TIMEZONE", default_camera.timezone),
        ),
        cameras=cameras,
    )
