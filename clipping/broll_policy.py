"""Pure policy helpers for validating and selecting Pexels B-roll assets."""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess


_GENERIC_QUERY_WORDS = {
    "business", "camera", "company", "education", "failure", "money", "office",
    "people", "success", "technology", "teamwork", "working", "work", "meeting",
    "laptop", "sad", "happy", "thinking", "handshake", "man", "woman",
}
_COMMAND_WORDS = {"bash", "cmd", "curl", "ffmpeg", "powershell", "python", "sh", "wget"}
_JOB_STATE = {}


def _text(value):
    return " ".join(str(value or "").split()).strip()


def _finite_float(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def normalize_broll_query(query):
    """Return a concise safe search query, or None for generic/unsafe input."""
    raw = str(query or "")
    if "\n" in raw or "\r" in raw:
        return None
    normalized = _text(raw)
    lowered = normalized.lower()
    if not normalized or len(normalized) > 120:
        return None
    if any(char in normalized for char in ("\n", "\r", "`", "$", "|", ";", "<", ">")):
        return None
    if "http://" in lowered or "https://" in lowered or lowered.startswith("www."):
        return None
    words = re.findall(r"[a-z0-9][a-z0-9'-]*", lowered)
    if len(words) < 2 or len(words) > 10:
        return None
    raw_tokens = normalized.split()
    if words[0] in _COMMAND_WORDS or any(token.startswith("-") for token in raw_tokens):
        return None
    if len([word for word in words if word not in _GENERIC_QUERY_WORDS]) < 2:
        return None
    return normalized


def simplify_broll_query(query):
    """Shorten a long query once without degrading it into a generic search."""
    normalized = normalize_broll_query(query)
    if normalized is None:
        return None
    words = normalized.split()
    if len(words) <= 4:
        return None
    simplified = normalize_broll_query(" ".join(words[:4]))
    return simplified if simplified and simplified != normalized else None


def normalize_visual_intent(intent):
    """Normalize a short description of what the stock visual illustrates."""
    normalized = _text(intent)
    if not normalized or len(normalized) > 180:
        return None
    words = normalized.split()
    if len(words) < 3 or len(words) > 24:
        return None
    if any(char in normalized for char in ("\n", "\r", "`", "$", "|", ";", "<", ">")):
        return None
    return normalized


def _best_video_file(video_files, target_ratio):
    usable = []
    for video_file in video_files if isinstance(video_files, list) else []:
        if not isinstance(video_file, dict) or video_file.get("file_type") != "video/mp4":
            continue
        link = str(video_file.get("link") or "")
        width = _finite_float(video_file.get("width"), 0.0)
        height = _finite_float(video_file.get("height"), 0.0)
        if not link.startswith("https://") or width < 480 or height < 480:
            continue
        aspect = width / height
        aspect_error = abs(math.log(max(aspect, 1e-6) / target_ratio))
        pixel_count = min(width * height, 1080 * 1920)
        quality_rank = 1 if str(video_file.get("quality", "")).lower() == "hd" else 0
        usable.append((aspect_error, -quality_rank, -pixel_count, video_file, int(width), int(height)))
    if not usable:
        return None
    aspect_error, _, _, video_file, width, height = min(usable, key=lambda item: item[:3])
    return {
        "file_url": video_file["link"],
        "width": width,
        "height": height,
        "aspect_error": aspect_error,
    }


def rank_pexels_candidates(videos, target_ratio, minimum_duration=0.0, used_ids=None, max_candidates=3):
    """Filter and rank existing Pexels results without making another API request."""
    ratio = _finite_float(target_ratio, None)
    min_duration = max(_finite_float(minimum_duration, 0.0), 0.0)
    if ratio is None or ratio <= 0 or not isinstance(videos, list):
        return []

    used_ids = used_ids or set()
    ranked = []
    for index, video in enumerate(videos):
        if not isinstance(video, dict):
            continue
        asset_id = video.get("id")
        if asset_id is None or asset_id in used_ids:
            continue
        duration = _finite_float(video.get("duration"), None)
        if duration is None or duration <= 0 or duration + 0.05 < min_duration:
            continue
        best_file = _best_video_file(video.get("video_files"), ratio)
        if best_file is None:
            continue

        duration_waste = max(duration - min_duration, 0.0) if min_duration else duration
        ranked.append((
            best_file["aspect_error"],
            duration_waste,
            -best_file["width"] * best_file["height"],
            index,
            {
                "asset_id": asset_id,
                "video_url": str(video.get("url") or ""),
                "duration": duration,
                **best_file,
            },
        ))

    ranked.sort(key=lambda item: item[:4])
    safe_limit = min(max(int(max_candidates), 1), 5)
    return [item[-1] for item in ranked[:safe_limit]]


def _session_key(output_dir):
    return os.path.normcase(os.path.abspath(output_dir or os.getcwd()))


def reset_broll_session(output_dir):
    """Start a job-scoped used-ID and asset-metadata session."""
    _JOB_STATE[_session_key(output_dir)] = {"used_ids": set(), "metadata": {}}


def _job_state(output_dir):
    return _JOB_STATE.setdefault(_session_key(output_dir), {"used_ids": set(), "metadata": {}})


def get_used_asset_ids(output_dir):
    return set(_job_state(output_dir)["used_ids"])


def mark_asset_used(output_dir, asset_id):
    _job_state(output_dir)["used_ids"].add(asset_id)


def record_asset_metadata(output_dir, output_path, metadata):
    path_key = os.path.normcase(os.path.abspath(output_path))
    _job_state(output_dir)["metadata"][path_key] = dict(metadata)


def get_asset_metadata(output_dir, output_path):
    path_key = os.path.normcase(os.path.abspath(output_path))
    data = _job_state(output_dir)["metadata"].get(path_key)
    return dict(data) if data else None


def cache_asset_path(output_dir, asset_id, file_url):
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(asset_id))[:80] or "asset"
    digest = hashlib.sha256(str(file_url).encode("utf-8")).hexdigest()[:10]
    return os.path.join(output_dir or os.getcwd(), "broll_cache", f"{safe_id}_{digest}.mp4")


def probe_broll_media(path, minimum_duration=0.0, ffprobe_path=None, timeout=15):
    """Return validated video metadata, or None for missing, corrupt, or short media."""
    if not path or not os.path.isfile(path):
        return None
    try:
        if os.path.getsize(path) <= 0:
            return None
    except OSError:
        return None

    probe = ffprobe_path or shutil.which("ffprobe")
    if not probe:
        return None
    try:
        result = subprocess.run(
            [probe, "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration",
             "-of", "json", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=timeout,
        )
        data = json.loads(result.stdout or "{}")
        stream = next((item for item in data.get("streams", []) if item.get("codec_type") == "video"), None)
        duration = _finite_float((data.get("format") or {}).get("duration"), None)
        width = int(_finite_float(stream.get("width"), 0.0)) if stream else 0
        height = int(_finite_float(stream.get("height"), 0.0)) if stream else 0
        min_duration = max(_finite_float(minimum_duration, 0.0), 0.0)
        if not stream or width <= 0 or height <= 0 or duration is None or duration <= 0:
            return None
        if duration + 0.05 < min_duration:
            return None
        return {"duration": duration, "width": width, "height": height}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
