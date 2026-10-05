"""Allowlisted, human-reviewable publishing package export."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from . import final_metadata, pacing


def _safe_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_video(path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not path or not os.path.exists(path):
        return {}
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,codec_name,sample_rate,channels",
             "-of", "json", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=20,
        )
        data = json.loads(result.stdout or "{}")
        video = next((item for item in data.get("streams", []) if item.get("codec_type") == "video"), {})
        audio = next((item for item in data.get("streams", []) if item.get("codec_type") == "audio"), {})
        return {
            "duration": float((data.get("format") or {}).get("duration") or 0.0),
            "width": video.get("width"),
            "height": video.get("height"),
            "video_codec": video.get("codec_name"),
            "audio_codec": audio.get("codec_name"),
            "sample_rate": audio.get("sample_rate"),
            "channels": audio.get("channels"),
        }
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}


def _fmt_srt(seconds):
    value = max(float(seconds or 0.0), 0.0)
    milliseconds = int(round(value * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_final_srt(row, source_segments=None):
    """Build portable final-output captions from source and AI timed words."""
    events = []
    prefix = float(row.get("hook_output_offset", 0.0) or 0.0)
    vo = row.get("voiceover_plan") if isinstance(row.get("voiceover_plan"), dict) else None
    vo_data = row.get("voiceover") if isinstance(row.get("voiceover"), dict) else {}
    vo_segments = vo_data.get("segments", []) if isinstance(vo_data, dict) else []
    if vo_segments:
        for segment in vo_segments:
            if not isinstance(segment, dict):
                continue
            text = " ".join(str(segment.get("text") or "").split())
            start = _safe_float(segment.get("start"))
            end = _safe_float(segment.get("end"))
            if text and start is not None and end is not None and end > start:
                events.append((prefix + start, prefix + end, text))

    final_prefix = prefix + float(row.get("vo_render_duration", 0.0) or 0.0)
    final_edit_map = row.get("pacing_final_map") or []
    edit_map = final_edit_map or row.get("pacing_map") or []
    for segment in source_segments if isinstance(source_segments, list) else []:
        if not isinstance(segment, dict):
            continue
        text = " ".join(str(segment.get("text") or "").split())
        start = _safe_float(segment.get("start"))
        end = _safe_float(segment.get("end"))
        if not text or start is None or end is None or end <= start:
            continue
        if edit_map:
            mapped_start = pacing.map_source_time(edit_map, start)
            mapped_end = pacing.map_source_time(edit_map, end)
        else:
            clip_start = float(row.get("start_time", 0.0) or 0.0)
            mapped_start = start - clip_start
            mapped_end = end - clip_start
        if mapped_start is None or mapped_end is None or mapped_end <= mapped_start:
            continue
        if final_edit_map:
            events.append((mapped_start, mapped_end, text))
        else:
            events.append((final_prefix + mapped_start, final_prefix + mapped_end, text))

    events.sort(key=lambda item: (item[0], item[1]))
    deduped = []
    seen = set()
    for event in events:
        key = (round(event[0], 3), round(event[1], 3), event[2])
        if key in seen:
            continue
        seen.add(key)
        if deduped and event[0] < deduped[-1][1] - 0.05 and event[2] == deduped[-1][2]:
            continue
        deduped.append(event)

    return "\n\n".join(
        f"{index}\n{_fmt_srt(start)} --> {_fmt_srt(end)}\n{text}"
        for index, (start, end, text) in enumerate(deduped, 1)
        if end > start
    ) + ("\n" if deduped else "")


def _safe_float(value):
    try:
        result = float(value)
        return result if result == result and abs(result) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_safe_json(value), handle, ensure_ascii=False, indent=2)


def _publish_text(row, metadata, source_info, qc):
    warnings = qc.get("warnings", []) if isinstance(qc, dict) else []
    status = str(row.get("quality_status") or qc.get("status") or "warning").upper()
    lines = [
        "=" * 50,
        f"{str(row.get('package_id', 'CLIP')).upper()}",
        "=" * 50,
        "",
        "STATUS",
        f"{status} — READY FOR HUMAN REVIEW" if status in {"PASS", "WARNING"} else "FAIL — DO NOT PUBLISH",
        "",
        "TITLE",
        metadata.get("title", ""),
        "",
        "DESCRIPTION",
        metadata.get("description", ""),
        "",
        "HASHTAGS",
        " ".join(metadata.get("hashtags", [])),
        "",
        "SOURCE",
        f"URL: {source_info.get('source_url') or 'not available'}",
        f"Selected source: {source_info.get('source_clip_start')} – {source_info.get('source_clip_end')}",
        "",
        "QC",
        status,
    ]
    if warnings:
        lines.extend(["", "WARNINGS", *[f"- {warning}" for warning in warnings]])
    lines.extend([
        "",
        "BEFORE PUBLISHING",
        "[ ] Watch the full video",
        "[ ] Verify title and thumbnail",
        "[ ] Verify captions, names, numbers, and context",
        "[ ] Confirm source rights/permission",
        "[ ] Confirm platform settings",
    ])
    return "\n".join(lines) + "\n"


def _package_clip(row, package_root, source_segments=None):
    clip_id = f"clip_{int(row.get('rank', 0) or 0):02d}"
    clip_dir = package_root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    video_path = row.get("video_path") or row.get("output_file")
    qc = row.get("quality_control") if isinstance(row.get("quality_control"), dict) else {}
    status = str(row.get("quality_status") or qc.get("status") or "fail").lower()
    package_status = "FAIL" if status == "fail" or not video_path or not os.path.exists(video_path) else ("WARNING" if status == "warning" else "PASS")
    row["package_id"] = clip_id

    files = {}
    if video_path and os.path.exists(video_path) and package_status != "FAIL":
        destination = clip_dir / "video.mp4"
        shutil.copy2(video_path, destination)
        files["video"] = "video.mp4"
    thumbnail_path = row.get("thumbnail_path")
    if thumbnail_path and os.path.exists(thumbnail_path):
        shutil.copy2(thumbnail_path, clip_dir / "thumbnail.jpg")
        files["thumbnail"] = "thumbnail.jpg"

    metadata = final_metadata.validate_metadata_package(row)
    metadata.update({"clip_id": clip_id, "status": package_status, "video_sha256": _sha256(video_path) if video_path and os.path.exists(video_path) else None})
    metadata_path = clip_dir / "metadata.json"
    _write_json(metadata_path, metadata)
    files["metadata"] = "metadata.json"

    source_info = {
        "source_url": row.get("source_url"),
        "source_title": row.get("source_title"),
        "source_clip_start": row.get("start_time"),
        "source_clip_end": row.get("end_time"),
        "final_media": _probe_video(video_path) if video_path else {},
    }
    _write_json(clip_dir / "source_info.json", source_info)
    files["source_info"] = "source_info.json"
    _write_json(clip_dir / "qc_report.json", qc)
    files["qc"] = "qc_report.json"
    _write_json(clip_dir / "edit_plan.json", {
        "hook": row.get("hook_plan", {}),
        "voiceover": row.get("voiceover_plan"),
        "timeline": row.get("voiceover_timeline", []),
        "pacing_map": row.get("pacing_final_map", row.get("pacing_map", [])),
        "broll_assets": row.get("broll_assets", []),
        "broll_fallbacks": row.get("broll_fallbacks", []),
    })
    files["edit_plan"] = "edit_plan.json"

    srt = build_final_srt(row, source_segments=source_segments)
    if srt:
        (clip_dir / "captions.srt").write_text(srt, encoding="utf-8")
        files["captions"] = "captions.srt"

    (clip_dir / "publish.txt").write_text(_publish_text(row, metadata, source_info, qc), encoding="utf-8")
    files["publish"] = "publish.txt"
    media = source_info["final_media"]
    manifest = {
        "clip_id": clip_id,
        "status": package_status,
        "files": files,
        "video_sha256": metadata.get("video_sha256"),
        "duration": media.get("duration"),
        "width": media.get("width"),
        "height": media.get("height"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(clip_dir / "manifest.json", manifest)
    return {"id": clip_id, "status": package_status, "title": metadata["title"], "path": str(clip_dir), "manifest": manifest}


def export_publishing_package(render_manifest, outputs_dir, source_segments=None, *, target_rank=None):
    """Create clean per-clip folders, batch summaries, and a validated ZIP."""
    package_root = Path(outputs_dir) / "publishing_package"
    if package_root.exists() and target_rank is None:
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)
    packages = []
    for row in render_manifest if isinstance(render_manifest, list) else []:
        row_rank = int(row.get("rank", 0) or 0)
        if target_rank is not None and row_rank != target_rank:
            clip_dir = package_root / f"clip_{row_rank:02d}"
            with (clip_dir / "manifest.json").open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            packages.append({
                "id": existing["clip_id"], "status": existing["status"],
                "title": row.get("title_inggris") or row.get("title") or "",
                "path": str(clip_dir), "manifest": existing,
            })
            continue
        if target_rank is not None:
            clip_dir = package_root / f"clip_{row_rank:02d}"
            if clip_dir.exists():
                shutil.rmtree(clip_dir)
        try:
            packages.append(_package_clip(row, package_root, source_segments=source_segments))
        except (OSError, TypeError, ValueError) as exc:
            if target_rank is not None:
                raise
            packages.append({"id": f"clip_{int(row.get('rank', 0) or 0):02d}", "status": "FAIL", "title": "", "error": str(exc)})

    summary = {
        "total": len(packages),
        "pass": sum(item["status"] == "PASS" for item in packages),
        "warning": sum(item["status"] == "WARNING" for item in packages),
        "fail": sum(item["status"] == "FAIL" for item in packages),
        "clips": [{key: item.get(key) for key in ("id", "status", "title")} for item in packages],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(package_root / "batch_summary.json", summary)
    lines = ["CLIPPING JOB SUMMARY", "", f"Total: {summary['total']}", f"PASS: {summary['pass']}", f"WARNING: {summary['warning']}", f"FAIL: {summary['fail']}", ""]
    for item in packages:
        lines.extend([f"{item['id']} — {item['status']}", item.get("title", ""), ""])
    lines.append("All clips require human review before publication.")
    (package_root / "batch_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    zip_path = Path(outputs_dir) / "publishing_package.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in package_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(package_root.parent))
    summary["package_root"] = str(package_root)
    summary["zip_path"] = str(zip_path)
    return summary
