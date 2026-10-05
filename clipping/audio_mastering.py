"""Conservative final loudness mastering for rendered Shorts."""

import json
import os
import re
import shutil
import subprocess

TARGET_I = -14.0
TARGET_TP = -1.0
TARGET_LRA = 11.0
AUDIO_BITRATE = "192k"
SAMPLE_RATE = 48000
CHANNELS = 2


def _finite(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result and abs(result) != float("inf") else default


def parse_loudnorm_json(stderr_text):
    """Extract the final loudnorm JSON object from FFmpeg stderr."""
    text = str(stderr_text or "")
    matches = re.findall(r"\{\s*\"input_i\".*?\}", text, flags=re.DOTALL)
    for raw in reversed(matches):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        required = {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"}
        if required.issubset(data):
            return data
    return None


def _run_loudnorm_analysis(video_path, ffmpeg_path="ffmpeg", timeout=120):
    command = [
        ffmpeg_path, "-hide_banner", "-i", video_path,
        "-af", f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:print_format=json",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return parse_loudnorm_json(result.stderr)


def measure_loudness(video_path, ffmpeg_path=None, timeout=120):
    """Measure final audio loudness; return None when FFmpeg cannot measure it."""
    ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg")
    if not ffmpeg_path or not video_path or not os.path.exists(video_path):
        return None
    data = _run_loudnorm_analysis(video_path, ffmpeg_path=ffmpeg_path, timeout=timeout)
    if not data:
        return None
    return {
        "integrated_lufs": _finite(data.get("input_i")),
        "true_peak_db": _finite(data.get("input_tp")),
        "loudness_range": _finite(data.get("input_lra")),
        "threshold_db": _finite(data.get("input_thresh")),
        "target_offset_db": _finite(data.get("target_offset")),
    }


def master_audio_in_place(video_path, output_path=None, timeout=180):
    """Normalize audio while copying video; preserve the original if mastering fails."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path or not video_path or not os.path.exists(video_path):
        return {"status": "skipped", "reason": "ffmpeg or input video unavailable"}

    output_path = output_path or f"{video_path}.mastered.mp4"
    measured = measure_loudness(video_path, ffmpeg_path=ffmpeg_path)
    if not measured:
        return {"status": "skipped", "reason": "loudness analysis unavailable"}

    loudnorm = (
        f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:"
        f"measured_I={measured['integrated_lufs']}:"
        f"measured_TP={measured['true_peak_db']}:"
        f"measured_LRA={measured['loudness_range']}:"
        f"measured_thresh={measured['threshold_db']}:"
        f"offset={measured['target_offset_db']}:linear=true:print_format=summary"
    )
    command = [
        ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y",
        "-i", video_path,
        "-map", "0:v:0", "-map", "0:a:0",
        "-map_metadata", "0",
        "-af", loudnorm,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "fallback", "reason": f"mastering process failed: {type(exc).__name__}"}

    if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        return {"status": "fallback", "reason": "mastered output was not created", "pre": measured}

    post = measure_loudness(output_path, ffmpeg_path=ffmpeg_path)
    if not post:
        os.remove(output_path)
        return {"status": "fallback", "reason": "mastered output could not be measured", "pre": measured}

    os.replace(output_path, video_path)
    return {
        "status": "mastered",
        "target_integrated_lufs": TARGET_I,
        "target_true_peak_db": TARGET_TP,
        "pre": measured,
        "post": post,
        "codec": "aac",
        "bitrate": AUDIO_BITRATE,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
    }
