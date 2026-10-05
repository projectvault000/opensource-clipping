"""Simple, file-backed checkpointing for resumable job runs.

The project already has long-running multi-stage workflows. This module gives
those stages a durable checkpoint state so a job can safely resume after an
interruption without restarting from zero.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_STATE_FILENAME = "job_state.json"


def _stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def deterministic_job_id(url: str | None = None, config: Mapping[str, Any] | None = None, *, salt: str = "") -> str:
    """Return a stable job id derived from the input URL and config.

    This is intentionally deterministic so the same job can be reloaded and
    resumed after a disconnect or interrupted run.
    """

    normalized: dict[str, Any] = {}
    if config is not None:
        if hasattr(config, "__dict__"):
            normalized.update(vars(config))
        else:
            normalized.update(dict(config))
    if url:
        normalized["url"] = url
    normalized["salt"] = salt

    payload = {}
    for key, value in normalized.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            payload[key] = dict(value)
        elif isinstance(value, (list, tuple, set)):
            payload[key] = sorted(str(item) for item in value)
        else:
            payload[key] = value

    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:16]
    return f"job_{digest}"


def job_id_for_config(cfg: Any) -> str:
    """Identify a CLI job without including secrets or runtime-specific paths."""
    excluded = {
        "base_dir", "outputs_dir", "font_dir", "cache_dir", "bgm_dir", "job_id",
        "url_list", "job_status", "clip_status", "approve", "reject",
        "needs_changes", "retry_clip", "review_note", "approved_export", "hf_token",
        "pexels_api_key", "yt_cookies",
    }
    settings = {
        key: value
        for key, value in vars(cfg).items()
        if key not in excluded
        and not key.startswith(("api_key_", "file_"))
        and isinstance(value, (str, int, float, bool, Mapping, list, tuple, set))
    }
    return deterministic_job_id(str(getattr(cfg, "url_youtube", "")), settings)


def prepare_cli_job(cfg: Any) -> None:
    """Place a CLI run in a stable job directory under the local outputs root."""
    if getattr(cfg, "job_id", None):
        return
    outputs_root = Path(cfg.base_dir) / "outputs"
    cfg.job_id = job_id_for_config(cfg)
    cfg.outputs_dir = str(outputs_root / cfg.job_id)
    cfg.file_video_asli = str(Path(cfg.outputs_dir) / "video_asli.mp4")
    cfg.cache_dir = str(outputs_root)


def render_manifest_complete(manifest: Any, expected_clips: int) -> bool:
    """Only reuse a completed render when every requested video is present."""
    if not isinstance(manifest, list) or len(manifest) != expected_clips:
        return False
    for row in manifest:
        if not isinstance(row, dict) or row.get("status") != "success":
            return False
        video_path = row.get("video_path")
        if not video_path or not os.path.isfile(video_path) or os.path.getsize(video_path) <= 0:
            return False
    return True


def build_resume_state_path(job_id: str, base_dir: str | os.PathLike[str] = "outputs") -> str:
    """Return the JSON state path for a resumable job."""

    return os.path.join(str(base_dir), str(job_id), DEFAULT_STATE_FILENAME)


def stage_fingerprint(*paths: str | os.PathLike[str]) -> str:
    """Compute a stable fingerprint for stage inputs/outputs.

    Any file that exists contributes its size and hash; missing files contribute
    their relative path to ensure invalidation is deterministic.
    """

    digest = hashlib.sha256()
    for raw_path in paths:
        path = str(raw_path)
        if not path:
            continue
        digest.update(path.encode("utf-8"))
        if os.path.exists(path):
            try:
                with open(path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(65536), b""):
                        digest.update(chunk)
            except OSError:
                digest.update(b"<unreadable>")
        else:
            digest.update(b"<missing>")
    return digest.hexdigest()


class JobCheckpoint:
    """Manage job-level checkpoints and stage status for resumable runs."""

    def __init__(
        self,
        job_dir: str | os.PathLike[str] | None = None,
        *,
        job_id: str | None = None,
        base_dir: str | os.PathLike[str] = "outputs",
        state_filename: str = DEFAULT_STATE_FILENAME,
    ) -> None:
        if job_dir is None:
            if job_id is None:
                raise ValueError("Either job_dir or job_id must be provided.")
            job_dir = os.path.join(str(base_dir), str(job_id))

        self.job_dir = Path(job_dir)
        self.job_id = job_id or self.job_dir.name if self.job_dir.name else "job_unknown"
        self.state_path = self.job_dir / state_filename
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self._state = self.load()

    @classmethod
    def from_cfg(cls, cfg: Any, *, state_filename: str = DEFAULT_STATE_FILENAME) -> "JobCheckpoint":
        """Create a checkpoint manager from the project config object."""

        job_id = getattr(cfg, "job_id", None)
        if job_id is None:
            job_id = deterministic_job_id(
                getattr(cfg, "url_youtube", None),
                getattr(cfg, "__dict__", None) or {},
            )
        job_dir = getattr(cfg, "outputs_dir", None)
        if job_dir is None:
            job_dir = os.path.join("outputs", job_id)
        return cls(job_dir=job_dir, job_id=job_id, state_filename=state_filename)

    def load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "job_id": self.job_id,
                "stages": {},
                "status": "new",
                "last_error": None,
                "updated_at": None,
                "created_at": None,
            }
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {
                "job_id": self.job_id,
                "stages": {},
                "status": "new",
                "last_error": None,
                "updated_at": None,
                "created_at": None,
            }
        if not isinstance(data, dict):
            return {
                "job_id": self.job_id,
                "stages": {},
                "status": "new",
                "last_error": None,
                "updated_at": None,
                "created_at": None,
            }
        data.setdefault("job_id", self.job_id)
        data.setdefault("stages", {})
        return data

    def save(self) -> None:
        self._state.setdefault("job_id", self.job_id)
        self._state.setdefault("stages", {})
        self.job_dir.mkdir(parents=True, exist_ok=True)
        temp_handle = tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            encoding="utf-8",
            dir=str(self.job_dir),
            suffix=".tmp",
        )
        try:
            json.dump(self._state, temp_handle, indent=2, ensure_ascii=False, sort_keys=True)
            temp_handle.flush()
            os.fsync(temp_handle.fileno())
        finally:
            temp_handle.close()
        os.replace(temp_handle.name, self.state_path)

    def _ensure_stage(self, stage_name: str) -> dict[str, Any]:
        stages = self._state.setdefault("stages", {})
        stage = stages.setdefault(stage_name, {})
        if not isinstance(stage, dict):
            stage = {}
            stages[stage_name] = stage
        return stage

    def mark_started(self, stage_name: str, **meta: Any) -> dict[str, Any]:
        stage = self._ensure_stage(stage_name)
        stage["status"] = "running"
        stage["attempts"] = int(stage.get("attempts", 0)) + 1
        stage["updated_at"] = datetime.now(timezone.utc).isoformat()
        for key, value in meta.items():
            if value is not None:
                stage[key] = value
        self._state["status"] = "running"
        self._state["updated_at"] = stage["updated_at"]
        self.save()
        return stage

    def mark_succeeded(self, stage_name: str, **meta: Any) -> dict[str, Any]:
        stage = self._ensure_stage(stage_name)
        stage["status"] = "succeeded"
        stage["updated_at"] = datetime.now(timezone.utc).isoformat()
        for key, value in meta.items():
            if value is not None:
                stage[key] = value
        if "fingerprint" not in stage:
            stage["fingerprint"] = stage_fingerprint(*[str(meta[k]) for k in sorted(meta) if isinstance(meta[k], (str, os.PathLike))])
        self._state["status"] = "running"
        self._state["updated_at"] = stage["updated_at"]
        self._state["last_error"] = None
        self.save()
        return stage

    def mark_failed(self, stage_name: str, error: str, **meta: Any) -> dict[str, Any]:
        stage = self._ensure_stage(stage_name)
        stage["status"] = "failed"
        stage["updated_at"] = datetime.now(timezone.utc).isoformat()
        stage["error"] = str(error)
        for key, value in meta.items():
            if value is not None:
                stage[key] = value
        self._state["status"] = "failed"
        self._state["updated_at"] = stage["updated_at"]
        self._state["last_error"] = str(error)
        self.save()
        return stage

    def is_complete(self, stage_name: str) -> bool:
        stage = self._state.get("stages", {}).get(stage_name, {})
        return stage.get("status") == "succeeded"

    def should_resume(self, stage_name: str, *, required_artifact: str | os.PathLike[str] | None = None) -> bool:
        """Return True when a stage should be retried or resumed."""

        stage = self._state.get("stages", {}).get(stage_name, {})
        if stage.get("status") == "succeeded":
            if required_artifact is None:
                return False
            return not os.path.exists(str(required_artifact))
        if required_artifact is not None and os.path.exists(str(required_artifact)):
            return False
        return True

    def stage_artifact(self, stage_name: str, *, key: str = "artifact_path") -> str | None:
        stage = self._state.get("stages", {}).get(stage_name, {})
        value = stage.get(key)
        if value in (None, ""):
            for candidate_key in ("artifact_path", "manifest_path", "output_file", "path"):
                value = stage.get(candidate_key)
                if value:
                    return str(value)
            return None
        return str(value)

    def status(self) -> str:
        if not self._state.get("stages"):
            return "new"
        if any(stage.get("status") == "failed" for stage in self._state["stages"].values()):
            return "failed"
        if all(stage.get("status") == "succeeded" for stage in self._state["stages"].values()):
            return "complete"
        return "running"

    def ensure_loaded(self) -> None:
        self._state = self.load()


CheckpointManager = JobCheckpoint
JobCheckpointState = JobCheckpoint


__all__ = [
    "JobCheckpoint",
    "CheckpointManager",
    "JobCheckpointState",
    "DEFAULT_STATE_FILENAME",
    "build_resume_state_path",
    "deterministic_job_id",
    "job_id_for_config",
    "prepare_cli_job",
    "render_manifest_complete",
    "stage_fingerprint",
]
