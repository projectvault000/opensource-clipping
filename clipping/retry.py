"""Saved inputs for a single-clip retry within an existing job."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import metadata


_JOB_ID = re.compile(r"job_[0-9a-f]{16}\Z")
_CLIP_ID = re.compile(r"clip_0*([1-9][0-9]*)\Z")
_SECRET_FIELDS = {"hf_token", "pexels_api_key", "yt_cookies"}


def validate_ids(job_id: str, clip_id: str) -> int:
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("Invalid job ID.")
    match = _CLIP_ID.fullmatch(clip_id)
    if not match or f"clip_{int(match.group(1)):02d}" != clip_id:
        raise ValueError("Invalid clip ID; expected clip_01, clip_02, etc.")
    return int(match.group(1))


def save_run_config(cfg: Any) -> None:
    path = Path(cfg.outputs_dir) / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    values = {
        key: value for key, value in vars(cfg).items()
        if not key.startswith("api_key_") and key not in _SECRET_FIELDS
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, ensure_ascii=False, indent=2)


def load_retry_config(outputs_root: str | os.PathLike[str], job_id: str, clip_id: str) -> tuple[SimpleNamespace, int]:
    rank = validate_ids(job_id, clip_id)
    job_dir = Path(outputs_root) / job_id
    with (job_dir / "run_config.json").open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict) or values.get("job_id") != job_id:
        raise ValueError("Saved run configuration does not match the requested job.")
    values["job_id"] = job_id
    values["outputs_dir"] = str(job_dir)
    values["file_video_asli"] = str(job_dir / "video_asli.mp4")
    values["cache_dir"] = str(outputs_root)
    values["api_key_gemini"] = os.environ.get("GOOGLE_API_KEY", "")
    values["api_key_nvidia"] = os.environ.get("NVIDIA_API_KEY", "")
    values["pexels_api_key"] = os.environ.get("PEXELS_API_KEY", "")
    values["hf_token"] = os.environ.get("HF_TOKEN", "")
    values["yt_cookies"] = None
    return SimpleNamespace(**values), rank


def normalize_retry_plan(
    saved_plan: list[dict], target_rank: int, source_duration: float | None,
    min_duration: float, max_duration: float,
) -> list[dict]:
    selected = [row for row in saved_plan if int(row.get("rank", 0)) == target_rank]
    if len(selected) != 1:
        raise ValueError(f"Saved plan must contain exactly one clip with rank {target_rank}.")
    normalized = metadata.normalize_and_validate(
        selected, source_duration=source_duration,
        min_duration=min_duration, max_duration=max_duration,
    )
    if len(normalized) != 1:
        raise ValueError(f"Saved plan for clip {target_rank} is invalid after validation.")
    normalized[0]["rank"] = target_rank
    return normalized
