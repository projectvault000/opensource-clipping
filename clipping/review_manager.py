from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoints import JobCheckpoint

VALID_PROCESSING = {"PENDING", "RUNNING", "COMPLETE", "FAILED", "INTERRUPTED"}
VALID_QC = {"PASS", "WARNING", "FAIL", "NOT_RUN"}
VALID_REVIEW = {"PENDING", "APPROVED", "REJECTED", "NEEDS_CHANGES", "STALE"}


def _normalize_status(value: Any, valid: set[str], default: str) -> str:
    text = str(value or "").strip().upper().replace("-", "_")
    if text in valid:
        return text
    if text in {"SUCCESS", "SUCCEEDED", "OK"}:
        if "COMPLETE" in valid:
            return "COMPLETE"
        if "PASS" in valid:
            return "PASS"
        return default
    if text in {"WARNINGS", "WARN"}:
        return "WARNING" if "WARNING" in valid else default
    return default


def _qc_value(row: dict[str, Any]) -> Any:
    quality_control = row.get("quality_control")
    return row.get("qc_status") or row.get("quality_status") or (
        quality_control.get("status") if isinstance(quality_control, dict) else None
    )


def _clip_id_for_row(row: dict[str, Any]) -> str:
    clip_id = str(row.get("clip_id") or "").strip()
    if clip_id:
        return clip_id
    rank = row.get("rank")
    if rank is None:
        return "clip_01"
    try:
        rank_num = int(rank)
    except (TypeError, ValueError):
        rank_num = 1
    return f"clip_{rank_num:02d}"


def _compute_review_fingerprint(row: dict[str, Any]) -> str:
    def file_version(raw_path: Any) -> Any:
        try:
            path = Path(raw_path)
            if path.is_file():
                stat = path.stat()
                return stat.st_size, stat.st_mtime_ns
            if path.is_dir():
                return [
                    (item.relative_to(path).as_posix(), item.stat().st_size, item.stat().st_mtime_ns)
                    for item in sorted(path.rglob("*")) if item.is_file()
                ]
        except (OSError, TypeError, ValueError):
            pass
        return None

    payload = {
        "video_path": row.get("video_path"),
        "video_version": file_version(row.get("video_path")),
        "thumbnail_path": row.get("thumbnail_path"),
        "thumbnail_version": file_version(row.get("thumbnail_path")),
        "package_path": row.get("package_path") or row.get("publishing_package_path"),
        "package_version": file_version(row.get("package_path") or row.get("publishing_package_path")),
        "title": row.get("title_inggris") or row.get("title") or row.get("youtube_title_final"),
        "description": row.get("youtube_description_final") or row.get("description"),
        "hashtags": row.get("hastag") or row.get("youtube_tags_final"),
        "quality_status": row.get("quality_status") or row.get("qc_status"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:32]


def _safe_manifest(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


class ReviewManager:
    """Persist and inspect human review state without creating a second job database."""

    def __init__(self, job_dir: str | os.PathLike[str]) -> None:
        self.job_dir = Path(job_dir)
        self.checkpoint = JobCheckpoint(job_dir=self.job_dir)
        self.manifest_path = self.job_dir / "render_manifest.json"

    def _load_state(self) -> dict[str, Any]:
        state = self.checkpoint.load()
        state.setdefault("review", {})
        state["review"].setdefault("clips", {})
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        self.checkpoint._state = state
        self.checkpoint.save()

    def _manifest_rows(self) -> list[dict[str, Any]]:
        return _safe_manifest(self.manifest_path)

    def _save_manifest(self, manifest: list[dict[str, Any]]) -> None:
        self.job_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

    def _state_for_clip(self, clip_id: str) -> dict[str, Any]:
        state = self._load_state()
        return state.setdefault("review", {}).setdefault("clips", {}).setdefault(clip_id, {})

    def sync_manifest(self, manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(manifest, list):
            manifest = []
        state = self._load_state()
        review_state = state.setdefault("review", {})
        clips = review_state.setdefault("clips", {})
        for row in manifest:
            if not isinstance(row, dict):
                continue
            clip_id = _clip_id_for_row(row)
            row["clip_id"] = clip_id
            lifecycle = row.get("processing_status") or row.get("status") or "COMPLETE"
            row["processing_status"] = _normalize_status(lifecycle, VALID_PROCESSING, "PENDING")
            qc_value = _qc_value(row)
            row["qc_status"] = _normalize_status(qc_value, VALID_QC, "NOT_RUN")
            base_review = clips.get(clip_id, {}).get("review_status") if isinstance(clips.get(clip_id), dict) else None
            previous_generation = int(clips.get(clip_id, {}).get("generation_version") or 1)
            current_generation = int(row.get("generation_version") or 1)
            new_generation = current_generation > previous_generation
            explicit_review = row.get("review_status")
            if new_generation:
                resolved_review = "PENDING"
            elif explicit_review is None:
                resolved_review = base_review or "PENDING"
            elif str(explicit_review).strip().upper() == "PENDING" and base_review in VALID_REVIEW and base_review != "PENDING":
                resolved_review = base_review
            else:
                resolved_review = explicit_review
            row["review_status"] = _normalize_status(resolved_review, VALID_REVIEW, "PENDING")
            row["review_note"] = "" if new_generation else row.get("review_note") or clips.get(clip_id, {}).get("review_note") or ""
            row["reviewed_at"] = None if new_generation else row.get("reviewed_at") or clips.get(clip_id, {}).get("reviewed_at")
            row["generation_version"] = current_generation
            stored_fp = None if new_generation else row.get("review_fingerprint") or clips.get(clip_id, {}).get("review_fingerprint")
            current_fp = _compute_review_fingerprint(row)
            if row["review_status"] == "APPROVED" and stored_fp and stored_fp != current_fp:
                row["review_status"] = "STALE"
                row["review_fingerprint"] = stored_fp
                row["stale_review"] = True
            else:
                row["review_fingerprint"] = stored_fp or current_fp
                row["stale_review"] = bool(row.get("stale_review")) or row["review_status"] == "STALE"
            if row["review_status"] == "APPROVED":
                row["review_fingerprint"] = current_fp
                row["stale_review"] = False
            clips[clip_id] = {
                "clip_id": clip_id,
                "processing_status": row["processing_status"],
                "qc_status": row["qc_status"],
                "review_status": row["review_status"],
                "review_note": row["review_note"],
                "reviewed_at": row.get("reviewed_at"),
                "generation_version": row["generation_version"],
                "review_fingerprint": row["review_fingerprint"],
                "stale_review": row.get("stale_review", False),
            }
        self._save_state(state)
        self._save_manifest(manifest)
        return manifest

    def set_review_status(self, clip_id: str, review_status: str, *, note: str = "", reviewed_at: str | None = None) -> dict[str, Any]:
        value = _normalize_status(review_status, VALID_REVIEW, "PENDING")
        manifest = self._manifest_rows()
        final_row = None
        for row in manifest:
            if _clip_id_for_row(row) != clip_id:
                continue
            final_row = row
            row["review_status"] = value
            row["review_note"] = note
            row["reviewed_at"] = reviewed_at or datetime.now(timezone.utc).isoformat()
            row["review_fingerprint"] = _compute_review_fingerprint(row)
            row["stale_review"] = False
            if value == "PENDING":
                row["reviewed_at"] = None
            break
        if final_row is None:
            final_row = {"clip_id": clip_id, "review_status": value, "review_note": note, "reviewed_at": reviewed_at or datetime.now(timezone.utc).isoformat(), "review_fingerprint": "", "stale_review": False}
            manifest.append(final_row)
        self._save_manifest(manifest)

        state = self._load_state()
        review = state.setdefault("review", {})
        clip_map = review.setdefault("clips", {})
        clip_state = clip_map.setdefault(clip_id, {})
        clip_state.update({
            "clip_id": clip_id,
            "review_status": value,
            "review_note": note,
            "reviewed_at": final_row.get("reviewed_at"),
            "review_fingerprint": final_row.get("review_fingerprint") or _compute_review_fingerprint(final_row),
            "stale_review": False,
        })
        self._save_state(state)
        return clip_state

    def get_manifest(self) -> list[dict[str, Any]]:
        return self._manifest_rows()

    def get_clip(self, clip_id: str) -> dict[str, Any]:
        manifest = self._manifest_rows()
        for row in manifest:
            if _clip_id_for_row(row) == clip_id:
                state = self._state_for_clip(clip_id)
                merged = {**state, **row}
                if merged.get("review_status") == "APPROVED":
                    current_fp = _compute_review_fingerprint(row)
                    stored_fp = merged.get("review_fingerprint") or row.get("review_fingerprint")
                    if stored_fp and stored_fp != current_fp:
                        merged["review_status"] = "STALE"
                        merged["stale_review"] = True
                return merged
        state = self._state_for_clip(clip_id)
        return {"clip_id": clip_id, "processing_status": "PENDING", "qc_status": "NOT_RUN", "review_status": state.get("review_status", "PENDING"), "review_note": state.get("review_note", ""), "reviewed_at": state.get("reviewed_at"), "stale_review": False}

    def get_review_summary(self) -> dict[str, Any]:
        manifest = self._manifest_rows()
        summary = {
            "total": len(manifest),
            "pending": 0,
            "approved": 0,
            "rejected": 0,
            "needs_changes": 0,
            "stale": 0,
            "failed": 0,
            "warning": 0,
            "pass": 0,
        }
        for row in manifest:
            review_status = _normalize_status(row.get("review_status") or row.get("human_review_status"), VALID_REVIEW, "PENDING")
            qc_status = _normalize_status(_qc_value(row), VALID_QC, "NOT_RUN")
            if review_status == "PENDING":
                summary["pending"] += 1
            elif review_status == "APPROVED":
                summary["approved"] += 1
            elif review_status == "REJECTED":
                summary["rejected"] += 1
            elif review_status == "NEEDS_CHANGES":
                summary["needs_changes"] += 1
            elif review_status == "STALE":
                summary["stale"] += 1
            if qc_status == "FAIL":
                summary["failed"] += 1
            elif qc_status == "WARNING":
                summary["warning"] += 1
            elif qc_status == "PASS":
                summary["pass"] += 1
        return summary

    def iter_review_queue(self) -> list[dict[str, Any]]:
        manifest = self._manifest_rows()
        queue = []
        for row in manifest:
            clip_id = _clip_id_for_row(row)
            review_status = _normalize_status(row.get("review_status"), VALID_REVIEW, "PENDING")
            if review_status in {"PENDING", "STALE", "NEEDS_CHANGES"}:
                queue.append({"clip_id": clip_id, "review_status": review_status, "qc_status": _normalize_status(row.get("qc_status") or row.get("quality_status"), VALID_QC, "NOT_RUN")})
        return queue

    def build_status_lines(self) -> str:
        manifest = self._manifest_rows()
        lines = [
            f"JOB: {self.checkpoint.job_id}",
            "",
            "SHARED",
            "Download        " + _status_label(self.checkpoint.load().get("stages", {}).get("download", {}).get("status"), "COMPLETE"),
            "Transcript      " + _status_label(self.checkpoint.load().get("stages", {}).get("transcribe", {}).get("status"), "COMPLETE"),
            "Selection       " + _status_label(self.checkpoint.load().get("stages", {}).get("render", {}).get("status"), "COMPLETE"),
            "",
            "CLIPS",
        ]
        for row in manifest:
            clip_id = _clip_id_for_row(row)
            qc = _normalize_status(row.get("qc_status") or row.get("quality_status"), VALID_QC, "NOT_RUN")
            review = _normalize_status(row.get("review_status"), VALID_REVIEW, "PENDING")
            processing = _normalize_status(row.get("processing_status") or row.get("status"), VALID_PROCESSING, "PENDING")
            lines.append(f"{clip_id}  {processing:<8}  {qc:<5}  {review}")
        return "\n".join(lines)

    def export_approved_set(self, *, manifest: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        rows = manifest or self._manifest_rows()
        rows = self.sync_manifest(rows)
        approved = build_approved_manifest(rows)
        if not approved:
            raise ValueError("No approved clips with passing QC are available for export.")
        output_dir = self.job_dir / "approved"
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "approved_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump([{"clip_id": item.get("clip_id"), "review_status": item.get("review_status"), "package_path": item.get("publishing_package_path") or item.get("package_path") or item.get("video_path")} for item in approved], handle, ensure_ascii=False, indent=2)
        archive_path = self.job_dir / "approved_clips.zip"
        if archive_path.exists():
            archive_path.unlink()
        job_root = self.job_dir.resolve()
        try:
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.write(manifest_path, "approved_manifest.json")
                for item in approved:
                    package_path = item.get("publishing_package_path") or item.get("package_path")
                    if package_path and os.path.exists(package_path):
                        package = Path(package_path).resolve()
                        if not package.is_relative_to(job_root):
                            raise ValueError(f"Package path is outside the job directory: {package}")
                        if os.path.isdir(package_path):
                            for root, _, files in os.walk(package_path):
                                for filename in files:
                                    file_path = Path(root, filename).resolve()
                                    if not file_path.is_relative_to(job_root):
                                        raise ValueError(f"Package file is outside the job directory: {file_path}")
                                    archive.write(file_path, file_path.relative_to(job_root).as_posix())
                        else:
                            archive.write(package, package.relative_to(job_root).as_posix())
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise
        with zipfile.ZipFile(archive_path) as archive:
            has_files = any(name != "approved_manifest.json" for name in archive.namelist())
        if not has_files:
            archive_path.unlink()
            raise ValueError("Approved clips have no package files to export.")
        return {"approved": approved, "manifest_path": str(manifest_path), "zip_path": str(archive_path)}


def build_approved_manifest(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    approved: list[dict[str, Any]] = []
    for row in manifest:
        if not isinstance(row, dict):
            continue
        qc_status = _normalize_status(_qc_value(row), VALID_QC, "NOT_RUN")
        review_status = _normalize_status(row.get("review_status") or row.get("human_review_status"), VALID_REVIEW, "PENDING")
        processing_status = _normalize_status(row.get("processing_status") or row.get("status"), VALID_PROCESSING, "PENDING")
        if processing_status != "COMPLETE":
            continue
        if qc_status not in {"PASS", "WARNING"}:
            continue
        if review_status == "APPROVED" and not row.get("stale_review"):
            approved.append(row)
    return approved


def _status_label(value: Any, default: str) -> str:
    status = str(value or default).upper()
    return status


__all__ = [
    "ReviewManager",
    "build_approved_manifest",
    "VALID_PROCESSING",
    "VALID_QC",
    "VALID_REVIEW",
]
