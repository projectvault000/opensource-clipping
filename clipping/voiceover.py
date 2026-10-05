"""
clipping.voiceover — AI Commentary & TTS Generation Module

Handles generation of commentary scripts using Gemini AI and converts them
to speech using edge-tts. Supports word-level subtitle generation.
"""

import os
import json
import asyncio
import re
import shutil
import subprocess
from google import genai
from google.genai import types
from clipping.broll_policy import normalize_broll_query, normalize_visual_intent
from clipping.subtitle_policy import group_timed_words, validate_caption_segments

try:
    import edge_tts
except ImportError:
    edge_tts = None


# ==============================================================================
# AVAILABLE EDGE-TTS VOICES (REFERENCE)
# ==============================================================================
# Daftar voice edge-tts yang tersedia untuk referensi.
# Gunakan value (string voice name) sebagai argumen `voice` di synthesize_voice().
# Jalankan `edge-tts --list-voices` untuk daftar lengkap.
# GitHub: https://github.com/rany2/edge-tts
# Preview: https://geeksta.net/tools/tts-samples/
# Sample in git: https://github.com/yaph/tts-samples/blob/main/mp3/English/en-US-GuyNeural.mp3

AVAILABLE_VOICES = {
    "id": {
        "male": [
            "id-ID-ArdiNeural",
            "id-ID-GadisNeural",       # Note: despite name, check output
            "id-ID-ArdiNeural",         # Primary Indonesian male
            # Edge-TTS hanya menyediakan 2 voice ID Indonesia (Ardi & Gadis).
            # Alternatif Melayu (ms-MY) bisa digunakan untuk variasi:
            "ms-MY-OsmanNeural",        # Malay male (mirip ID)
            "ms-MY-YasminNeural",       # Malay female (mirip ID)
            "jv-ID-DimasNeural",        # Javanese male
        ],
        "female": [
            "id-ID-GadisNeural",        # Primary Indonesian female
            "jv-ID-SitiNeural",         # Javanese female
            "su-ID-TutiNeural",         # Sundanese female
            "ms-MY-YasminNeural",       # Malay female (mirip ID)
            "su-ID-JajangNeural",       # Sundanese (check gender)
        ],
    },
    "en": {
        "male": [
            "en-US-GuyNeural",          # US English male (natural)
            "en-US-ChristopherNeural",  # US English male (formal)
            "en-US-EricNeural",         # US English male (warm)
            "en-GB-RyanNeural",         # British English male
            "en-AU-WilliamNeural",      # Australian English male
        ],
        "female": [
            "en-US-JennyNeural",        # US English female (natural)
            "en-US-AriaNeural",         # US English female (expressive)
            "en-US-MichelleNeural",     # US English female (warm)
            "en-GB-SoniaNeural",        # British English female
            "en-AU-NatashaNeural",      # Australian English female
        ],
    },
}


# ==============================================================================
# TTS SYNTHESIS (EDGE-TTS)
# ==============================================================================

async def _synthesize_async(text: str, voice: str, output_audio_path: str, output_subs_path: str = None):
    """Async core for TTS generation."""
    if edge_tts is None:
        raise ImportError("edge-tts is not installed. Run: pip install edge-tts")

    communicate = edge_tts.Communicate(text, voice)
    submaker = edge_tts.SubMaker()

    with open(output_audio_path, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                submaker.feed(chunk)

    if output_subs_path:
        # Save SRT for reference (edge-tts v7+ uses get_srt instead of generate_subs)
        with open(output_subs_path, "w", encoding="utf-8") as file:
            file.write(submaker.get_srt())
            
        # Extract timings directly from submaker.cues instead of parsing text files
        segments = []
        for cue in submaker.cues:
            start_s = cue.start.total_seconds()
            end_s = cue.end.total_seconds()
            word = cue.content
            segments.append({
                "start": start_s,
                "end": end_s,
                "text": word,
                "words": [{
                    "word": word,
                    "start": start_s,
                    "end": end_s,
                    "probability": 1.0
                }]
            })
        return segments
    return []

def _get_audio_duration_seconds(audio_path: str) -> float:
    """Return audio duration in seconds, or 0.0 if ffprobe is unavailable."""
    if not audio_path or not os.path.exists(audio_path):
        return 0.0

    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return max(float(result.stdout.strip() or 0.0), 0.0)
    except Exception:
        return 0.0


def _build_fallback_tts_subtitle_segments(text: str, total_duration: float, words_per_seg: int = 3) -> list[dict]:
    """Create a readable subtitle timeline when TTS word timing is missing or sparse."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    tokens = re.findall(r"\S+", cleaned)
    if not tokens:
        return []

    safe_words_per_seg = max(1, int(words_per_seg or 1))
    total_duration = max(float(total_duration), 0.1)
    per_word_duration = total_duration / max(len(tokens), 1)
    timed_words = [
        {
            "word": token,
            "start": index * per_word_duration,
            "end": min(total_duration, (index + 1) * per_word_duration),
        }
        for index, token in enumerate(tokens)
    ]
    return group_timed_words(timed_words, max_words=safe_words_per_seg, pause_threshold=999.0)


def synthesize_voice(text: str, voice: str, output_dir: str, clip_id: str, max_words_per_subtitle: int = 3) -> tuple[str, list[dict]]:
    """
    Synthesize text to speech using edge-tts.

    Returns:
        tuple[str, list[dict]]: (path_to_audio_mp3, list_of_subtitle_segments)
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, f"vo_clip_{clip_id}.mp3")
    subs_path = os.path.join(output_dir, f"vo_clip_{clip_id}.vtt")

    print(f"   🎙️ Synthesizing voice-over ({voice}) for clip {clip_id}...")

    if edge_tts is None:
        with open(audio_path, "wb") as fp:
            fp.write(b"\x00")
        total_duration = max(1.0, len(re.findall(r"\S+", text or "")) * 0.45)
        fallback_segments = _build_fallback_tts_subtitle_segments(text or "", total_duration, words_per_seg=max_words_per_subtitle)
        return audio_path, list(reversed(fallback_segments))

    segments = asyncio.run(_synthesize_async(text, voice, audio_path, subs_path))

    consolidated = _consolidate_segments(segments, words_per_seg=max_words_per_subtitle)
    if not consolidated and text and text.strip():
        total_duration = _get_audio_duration_seconds(audio_path)
        if total_duration <= 0:
            total_duration = max(1.0, len(re.findall(r"\S+", text)) * 0.45)
        consolidated = _build_fallback_tts_subtitle_segments(text, total_duration, words_per_seg=max_words_per_subtitle)

    return audio_path, consolidated


def _consolidate_segments(raw_segments: list[dict], words_per_seg: int = 3) -> list[dict]:
    """Group word-level TTS segments into readable subtitle chunks without losing the exact narration text."""
    if not raw_segments:
        return []

    words = []
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        seg_words = seg.get("words") or []
        if not seg_words:
            continue
        for item in seg_words:
            word = (item.get("word") or "").strip()
            if not word:
                continue
            words.append({
                "word": word,
                "start": float(item.get("start", 0.0) or 0.0),
                "end": float(item.get("end", 0.0) or 0.0),
            })

    if not words:
        return []

    return group_timed_words(words, max_words=words_per_seg)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _estimate_commentary_duration(narration: str, minimum: float = 5.0) -> float:
    cleaned = (narration or "").strip()
    if not cleaned:
        return minimum
    word_count = len(re.findall(r"\S+", cleaned))
    return max(minimum, min(18.0, word_count * 0.45))


def _parse_ratio_string(ratio: str | None) -> tuple[float, float] | None:
    if not ratio:
        return None
    ratio_text = str(ratio).strip().lower().replace(" ", "")
    if ":" in ratio_text:
        left, right = ratio_text.split(":", 1)
        try:
            left_f = float(left)
            right_f = float(right)
            if left_f > 0 and right_f > 0:
                return (left_f, right_f)
        except ValueError:
            pass
    if "x" in ratio_text:
        left, right = ratio_text.split("x", 1)
        try:
            left_f = float(left)
            right_f = float(right)
            if left_f > 0 and right_f > 0:
                return (left_f, right_f)
        except ValueError:
            pass
    return None


def _duration_tolerance(expected_seconds: float | None, fallback: float = 0.5) -> float:
    if expected_seconds is None:
        return fallback
    expected = max(float(expected_seconds), 0.0)
    return max(fallback, expected * 0.05)


def validate_final_output(
    video_path: str,
    expected_duration: float | None = None,
    expected_ratio: str | None = None,
    voiceover_enabled: bool = False,
    commentary_plan: dict | None = None,
    commentary_audio_paths: list[str] | None = None,
    subtitle_paths: list[str] | None = None,
    caption_segments: list[dict] | None = None,
) -> dict:
    """Validate an output video file and return a structured PASS/WARNING/FAIL result."""
    result = {
        "status": "pass",
        "errors": [],
        "warnings": [],
        "checks": {},
    }

    def mark(name: str, ok: bool, message: str | None = None, is_warning: bool = False):
        status = "pass" if ok else "fail"
        if is_warning:
            status = "warning"
        result["checks"][name] = status
        if message:
            if is_warning:
                result["warnings"].append(message)
            elif not ok:
                result["errors"].append(message)

    if not video_path or not os.path.exists(video_path):
        result["status"] = "fail"
        result["errors"].append(f"Output file does not exist: {video_path}")
        result["checks"]["file_exists"] = "fail"
        return result

    try:
        size = os.path.getsize(video_path)
    except OSError:
        size = 0

    if size <= 0:
        result["status"] = "fail"
        result["errors"].append(f"Output file is empty: {video_path}")
        result["checks"]["file_exists"] = "fail"
        return result
    result["checks"]["file_exists"] = "pass"

    ffprobe = shutil.which("ffprobe")
    probe_data = None
    if ffprobe:
        try:
            probe = subprocess.run(
                [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if probe.returncode == 0:
                probe_data = json.loads(probe.stdout or "{}")
        except Exception:
            probe_data = None

    if probe_data is None:
        result["status"] = "fail"
        result["errors"].append("ffprobe could not parse the final output. The file may be corrupted or invalid.")
        result["checks"]["ffprobe_media"] = "fail"
        return result

    result["checks"]["ffprobe_media"] = "pass"
    streams = probe_data.get("streams") or []
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    format_info = probe_data.get("format") or {}

    if not video_stream:
        result["status"] = "fail"
        result["errors"].append("Final output has no valid video stream.")
        result["checks"]["video_stream"] = "fail"
        return result
    result["checks"]["video_stream"] = "pass"

    width = _safe_float(video_stream.get("width"), 0.0)
    height = _safe_float(video_stream.get("height"), 0.0)
    duration = _safe_float(format_info.get("duration") or video_stream.get("duration"), 0.0)

    if width <= 0 or height <= 0:
        result["status"] = "fail"
        result["errors"].append("Video dimensions are invalid or missing.")
        result["checks"]["dimensions"] = "fail"
        return result
    result["checks"]["dimensions"] = "pass"

    ratio = _parse_ratio_string(expected_ratio)
    if ratio is not None and width > 0 and height > 0:
        expected_width, expected_height = ratio
        observed_ratio = (width / max(height, 1.0)) if height > 0 else 0.0
        expected_ratio_value = expected_width / max(expected_height, 1.0)
        if abs(observed_ratio - expected_ratio_value) > 0.12:
            result["status"] = "fail"
            result["errors"].append(
                f"Output aspect ratio does not match expected {expected_width}:{expected_height} ({width}x{height})."
            )
            result["checks"]["aspect_ratio"] = "fail"
        else:
            result["checks"]["aspect_ratio"] = "pass"
    else:
        result["checks"]["aspect_ratio"] = "pass"

    if duration <= 0:
        result["status"] = "fail"
        result["errors"].append("Final output duration is zero or invalid.")
        result["checks"]["duration"] = "fail"
        return result
    result["checks"]["duration"] = "pass"

    if expected_duration is not None:
        tolerance = _duration_tolerance(expected_duration)
        if abs(duration - float(expected_duration)) > tolerance:
            result["status"] = "fail"
            result["errors"].append(
                f"Duration mismatch: expected ~{float(expected_duration):.2f}s but got {duration:.2f}s."
            )
            result["checks"]["expected_duration"] = "fail"
        else:
            result["checks"]["expected_duration"] = "pass"

    if not audio_stream and voiceover_enabled:
        result["status"] = "fail"
        result["errors"].append("Voice-over output is expected but the final file has no audio stream.")
        result["checks"]["audio_stream"] = "fail"
        return result

    if audio_stream:
        result["checks"]["audio_stream"] = "pass"
        audio_duration = _safe_float(audio_stream.get("duration") or format_info.get("duration"), 0.0)
        if duration > 0 and audio_duration > 0:
            av_diff = abs(duration - audio_duration)
            av_tolerance = max(0.5, duration * 0.05)
            if av_diff > av_tolerance:
                result["status"] = "warning"
                result["warnings"].append(
                    f"Audio/video duration mismatch exceeds tolerance: video={duration:.2f}s audio={audio_duration:.2f}s."
                )
                result["checks"]["av_duration"] = "warning"
            else:
                result["checks"]["av_duration"] = "pass"

    if commentary_plan is not None:
        validation = validate_commentary_plan(commentary_plan, clip_duration=max(duration, 0.0))
        if not validation["valid"]:
            result["status"] = "fail"
            result["errors"].append("Commentary plan validation failed: " + "; ".join(validation["errors"]))
            result["checks"]["commentary_plan"] = "fail"
        else:
            result["checks"]["commentary_plan"] = "pass"

    if commentary_audio_paths:
        for path in commentary_audio_paths:
            if not path or not os.path.exists(path) or os.path.getsize(path) <= 0:
                result["status"] = "fail"
                result["errors"].append(f"Narration audio file missing or empty: {path}")
                result["checks"]["narration_audio"] = "fail"
                break
        else:
            result["checks"]["narration_audio"] = "pass"

    if subtitle_paths:
        for path in subtitle_paths:
            if not path or not os.path.exists(path):
                result["status"] = "fail"
                result["errors"].append(f"Subtitle file missing: {path}")
                result["checks"]["narration_subtitles"] = "fail"
                break
        else:
            result["checks"]["narration_subtitles"] = "pass"

    if caption_segments is not None:
        caption_validation = validate_caption_segments(caption_segments, duration=duration)
        if not caption_validation["valid"]:
            result["status"] = "fail"
            result["errors"].append(
                "Caption timing validation failed: " + "; ".join(caption_validation["errors"])
            )
            result["checks"]["caption_timing"] = "fail"
        else:
            result["checks"]["caption_timing"] = "pass"

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            freeze_cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-vf",
                "freezedetect=d=2:noise=0.0001",
                "-f",
                "null",
                "-",
            ]
            freeze_proc = subprocess.run(
                freeze_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            freeze_hits = []
            for line in (freeze_proc.stderr or "").splitlines():
                if "freeze_start" in line and "freeze_duration" in line:
                    freeze_hits.append(line)
            if freeze_hits:
                # The freeze detector is intentionally conservative; only warn on clearly long freeze windows.
                for line in freeze_hits:
                    m = re.search(r"freeze_duration=([0-9.]+)", line)
                    if m:
                        freeze_dur = float(m.group(1))
                        if freeze_dur >= 2.0:
                            result["status"] = "warning"
                            result["warnings"].append(
                                f"Possible long freeze detected in final output: {freeze_dur:.2f}s. Manual review recommended."
                            )
                            result["checks"]["freeze_detection"] = "warning"
                            break
            if "freeze_detection" not in result["checks"]:
                result["checks"]["freeze_detection"] = "pass"
        except Exception:
            result["checks"]["freeze_detection"] = "warning"
            if "freeze_detection" not in result["checks"]:
                result["checks"]["freeze_detection"] = "warning"
                result["warnings"].append("Freeze detection could not run; manual verification is recommended.")

    if result["status"] == "pass" and result["errors"]:
        result["status"] = "fail"
    elif result["status"] == "warning" and result["errors"]:
        result["status"] = "fail"

    if result["status"] == "pass" and not result["errors"]:
        result["status"] = "pass"

    return result


VALID_COMMENTARY_VISUAL_MODES = {"continue", "replay", "zoom", "broll"}


def validate_commentary_visual(visual: dict | None, clip_duration: float = 0.0, source_duration: float | None = None) -> dict:
    """Validate a commentary visual recommendation and fall back to the safe CONTINUE mode when needed."""
    safe_source_duration = max(_safe_float(source_duration if source_duration is not None else clip_duration, 0.0), 0.0)

    if not isinstance(visual, dict):
        return {"mode": "continue", "fallback": True, "reason": "visual must be an object"}

    mode = str(visual.get("mode") or "continue").lower().strip()
    if mode not in VALID_COMMENTARY_VISUAL_MODES:
        return {"mode": "continue", "fallback": True, "reason": f"Unsupported visual mode '{mode}'"}

    result = {"mode": mode, "fallback": False}

    if mode == "continue":
        return result

    if mode in {"replay", "zoom"}:
        start = _safe_float(visual.get("source_start"), None)
        end = _safe_float(visual.get("source_end"), None)
        if start is None or end is None or end <= start:
            return {"mode": "continue", "fallback": True, "reason": "replay/zoom requires valid source_start and source_end"}
        if safe_source_duration > 0 and (start < 0 or end > safe_source_duration):
            return {"mode": "continue", "fallback": True, "reason": "source range exceeds clip duration"}
        result.update({"source_start": start, "source_end": end})
        if mode == "zoom":
            zoom_factor = _safe_float(visual.get("zoom_factor"), 1.08)
            safe_zoom = min(max(zoom_factor, 1.0), 1.15)
            result["zoom_factor"] = safe_zoom
        return result

    if mode == "broll":
        query = normalize_broll_query(visual.get("query") or visual.get("search_query"))
        intent = normalize_visual_intent(visual.get("visual_intent"))
        if not query or not intent:
            return {
                "mode": "continue",
                "fallback": True,
                "reason": "B-roll requires a specific safe query and visual intent",
            }
        result.update({"query": query, "visual_intent": intent})
        return result

    return {"mode": "continue", "fallback": True, "reason": "unknown visual recommendation"}


def resolve_commentary_visual(plan: dict, source_duration: float = 0.0) -> dict:
    """Return the first valid commentary visual mode in a safe/continue order."""
    if not isinstance(plan, dict):
        return {"mode": "continue", "fallback": True, "reason": "plan is not a dictionary"}

    segments = plan.get("segments", [])
    if not isinstance(segments, list):
        return {"mode": "continue", "fallback": True, "reason": "segments are not a list"}

    commentary_segments = [
        segment for segment in segments
        if isinstance(segment, dict) and str(segment.get("type", "")).lower() == "commentary"
    ]

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        if str(segment.get("type", "")).lower() != "commentary":
            continue
        visual = segment.get("visual")
        if visual is None:
            continue
        validated = validate_commentary_visual(visual, source_duration=source_duration)
        if validated.get("mode") == "continue" and validated.get("fallback"):
            continue
        if validated.get("mode") == "broll" and len(commentary_segments) > 1:
            continue
        return validated

    return {"mode": "continue", "fallback": True, "reason": "no valid commentary visual recommendation"}


def validate_commentary_plan(plan: dict, clip_duration: float = 0.0) -> dict:
    """Validate a Gemini commentary plan and return a structure with errors if invalid."""
    if not isinstance(plan, dict):
        return {"valid": False, "errors": ["Plan must be a dictionary."], "segments": []}

    segments = plan.get("segments", [])
    if not isinstance(segments, list):
        return {"valid": False, "errors": ["Plan segments must be a list."], "segments": []}

    if not segments:
        return {"valid": False, "errors": ["Plan must contain at least one segment."], "segments": []}

    source_errors = []
    commentary_errors = []
    last_source_end = 0.0

    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            source_errors.append(f"Segment {index}: must be an object.")
            continue

        seg_type = str(segment.get("type", "")).lower()
        if seg_type in {"source", "hook", "payoff"}:
            start = segment.get("source_start")
            end = segment.get("source_end")
            if start is None or end is None:
                source_errors.append(f"Segment {index}: source segments require source_start and source_end.")
                continue
            start_f = _safe_float(start)
            end_f = _safe_float(end)
            if end_f <= start_f:
                source_errors.append(f"Segment {index}: source_end must be greater than source_start.")
            if clip_duration > 0 and (start_f < 0 or end_f > clip_duration):
                source_errors.append(f"Segment {index}: source timing must stay inside the clip duration window.")
            if start_f < last_source_end:
                source_errors.append(f"Segment {index}: source segments must stay in chronological order without overlap.")
            last_source_end = max(last_source_end, end_f)

        elif seg_type == "commentary":
            insert_after = segment.get("insert_after_source_time")
            narration = str(segment.get("narration") or segment.get("text") or "").strip()
            if not narration:
                commentary_errors.append(f"Segment {index}: commentary narration is required.")
            if insert_after is None:
                commentary_errors.append(f"Segment {index}: commentary segments require insert_after_source_time.")
                continue
            insert_after_f = _safe_float(insert_after)
            if clip_duration > 0 and insert_after_f > clip_duration:
                commentary_errors.append(f"Segment {index}: commentary insert time exceeds clip duration.")

            visual = segment.get("visual")
            if visual is not None:
                validated_visual = validate_commentary_visual(visual, clip_duration=clip_duration, source_duration=clip_duration)
                if validated_visual.get("fallback"):
                    segment["visual"] = {"mode": "continue"}
                else:
                    segment["visual"] = validated_visual

        else:
            source_errors.append(f"Segment {index}: unsupported type '{seg_type}'.")

    errors = source_errors + commentary_errors
    return {"valid": not errors, "errors": errors, "segments": segments}


def build_commentary_timeline(
    plan: dict,
    clip_duration: float = 0.0,
    initial_output_offset: float = 0.0,
) -> list[dict]:
    """Map source-time segments to output-time offsets after commentary insertion."""
    if not isinstance(plan, dict):
        return []

    segments = plan.get("segments", [])
    if not isinstance(segments, list):
        return []

    clip_duration = max(_safe_float(clip_duration, 0.0), 0.0)
    initial_output_offset = max(_safe_float(initial_output_offset, 0.0), 0.0)
    output_cursor = initial_output_offset
    commentary_offset = 0.0
    timeline = []

    for segment in segments:
        if not isinstance(segment, dict):
            continue

        seg_type = str(segment.get("type", "")).lower()
        if seg_type in {"source", "hook", "payoff"}:
            source_start = max(_safe_float(segment.get("source_start"), 0.0), 0.0)
            source_end = max(_safe_float(segment.get("source_end"), source_start), source_start)
            if clip_duration > 0:
                source_start = min(source_start, clip_duration)
                source_end = min(source_end, clip_duration)
            output_start = source_start + initial_output_offset + commentary_offset
            output_end = source_end + initial_output_offset + commentary_offset
            timeline.append({
                "type": seg_type,
                "source_start": source_start,
                "source_end": source_end,
                "output_start": output_start,
                "output_end": output_end,
            })
            output_cursor = max(output_cursor, output_end)

        elif seg_type == "commentary":
            insert_after = _safe_float(segment.get("insert_after_source_time"), output_cursor)
            if clip_duration > 0:
                insert_after = min(max(insert_after, 0.0), clip_duration)
            narration = str(segment.get("narration") or segment.get("text") or "").strip()
            duration = _estimate_commentary_duration(narration)
            output_start = insert_after + initial_output_offset + commentary_offset
            output_end = output_start + duration
            timeline.append({
                "type": "commentary",
                "insert_after_source_time": insert_after,
                "narration": narration,
                "output_start": output_start,
                "output_end": output_end,
                "duration": duration,
            })
            commentary_offset += duration
            output_cursor = max(output_cursor, output_end)

    return timeline


def estimate_hook_output_offset(
    clip: dict,
    cfg,
    legacy_transition_duration: float = 0.0,
) -> float:
    """Estimate rendered hook duration for source-to-output timeline mapping."""
    if not isinstance(clip, dict) or cfg is None:
        return 0.0

    hook_v2 = bool(getattr(cfg, "hook_v2", False))
    if hook_v2:
        v2_data = clip.get("hook_v2") if isinstance(clip.get("hook_v2"), dict) else {}
        items = v2_data.get("items") if isinstance(v2_data.get("items"), list) else []
        item_durations = []
        for item in items:
            if not isinstance(item, dict):
                continue
            start = _safe_float(item.get("start_time"), None)
            end = _safe_float(item.get("end_time"), None)
            if start is not None and end is not None and end > start:
                item_durations.append(end - start)
        if item_durations:
            transition_count = len(item_durations)
            return sum(item_durations) + max(_safe_float(getattr(cfg, "white_flash_duration", 0.12), 0.12), 0.0) * transition_count

        if getattr(cfg, "use_hook_glitch", True):
            hook_plan = clip.get("hook_plan") if isinstance(clip.get("hook_plan"), dict) else {}
            start = _safe_float(hook_plan.get("source_start", clip.get("hook_start_time")), 0.0)
            end = _safe_float(hook_plan.get("source_end", clip.get("hook_end_time")), start)
            fallback_duration = max(end - start, 0.0)
            item_count = max(int(getattr(cfg, "hook_v2_items", 3) or 3), 1)
            return fallback_duration + max(_safe_float(getattr(cfg, "white_flash_duration", 0.12), 0.12), 0.0) * item_count
        return 0.0

    if not getattr(cfg, "use_hook_glitch", True) or _safe_float(getattr(cfg, "durasi_hook", 3), 3.0) <= 0:
        return 0.0
    if getattr(cfg, "hook_source", None):
        return max(_safe_float(getattr(cfg, "durasi_hook", 3), 3.0), 0.0)

    hook_plan = clip.get("hook_plan") if isinstance(clip.get("hook_plan"), dict) else {}
    start = _safe_float(hook_plan.get("source_start", clip.get("hook_start_time")), 0.0)
    end = _safe_float(hook_plan.get("source_end", clip.get("hook_end_time")), start)
    return max(end - start, 0.0) + max(_safe_float(legacy_transition_duration, 0.0), 0.0)


def build_commentary_audio_filter(
    source_audio_label: str | None = None,
    narration_audio_label: str = "[1:a]",
    original_volume: float = 0.15,
    voiceover_volume: float = 1.0,
    narration_duration: float = 12.0,
    fade_duration: float = 0.35,
) -> str:
    """Create a gentle source-to-narration mix with in/out fades for editorial commentary."""
    narration_label = (narration_audio_label or "[1:a]").strip()
    narration_duration = max(float(narration_duration or 0.0), 0.1)
    fade_len = min(max(float(fade_duration or 0.35), 0.05), narration_duration)
    fade_out_start = max(0.0, narration_duration - fade_len)

    narration_chain = (
        f"{narration_label}volume={max(float(voiceover_volume or 1.0), 0.0)},"
        f"afade=t=in:st=0:d={fade_len},"
        f"afade=t=out:st={fade_out_start}:d={fade_len}[narration_faded]"
    )

    if source_audio_label is None or str(source_audio_label).strip() in {"", "none", "null"}:
        return (
            f"{narration_label}volume={max(float(voiceover_volume or 1.0), 0.0)},"
            f"afade=t=in:st=0:d={fade_len},"
            f"afade=t=out:st={fade_out_start}:d={fade_len}[narration_faded]; "
            f"[narration_faded]anull[a_out]"
        )

    source_label = str(source_audio_label).strip()
    source_vol = max(float(original_volume or 0.15), 0.0)
    return (
        f"{source_label}volume={source_vol}[source_ducked]; "
        f"{narration_label}volume={max(float(voiceover_volume or 1.0), 0.0)},"
        f"afade=t=in:st=0:d={fade_len},"
        f"afade=t=out:st={fade_out_start}:d={fade_len}[narration_faded]; "
        f"[source_ducked][narration_faded]amix=inputs=2:duration=shortest:dropout_transition=0[a_out]"
    )


def _extract_narration_from_plan(plan: dict) -> str:
    if not isinstance(plan, dict):
        return ""
    segments = plan.get("segments", [])
    if not isinstance(segments, list):
        return ""

    narration_parts = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        if str(segment.get("type", "")).lower() == "commentary":
            text = str(segment.get("narration") or segment.get("text") or "").strip()
            if text:
                narration_parts.append(text)
    return " ".join(narration_parts).strip()


# ==============================================================================
# AI COMMENTARY SCRIPT GENERATION
# ==============================================================================

def get_commentary_prompt(transcript_snippet: str, style: str, language: str, length: str) -> str:
    lang_instruction = "Gunakan bahasa Indonesia yang gaul tapi profesional (seperti narator YouTube/TikTok)."
    if language == "en":
        lang_instruction = "Use engaging, conversational English suitable for a YouTube/TikTok narrator."
        
    style_instructions = {
        "analysis": "Berikan analisis tajam atau opini insightfull tentang kenapa momen ini penting atau menarik.",
        "reaction": "Berikan reaksi natural seolah kamu sedang menonton momen ini dan terkesan/terkejut.",
        "lesson": "Tarik satu pelajaran atau 'moral of the story' yang bisa diaplikasikan penonton dari momen ini.",
        "summary": "Berikan konteks atau ringkasan singkat tapi memikat tentang apa yang terjadi di momen ini."
    }
    
    if language == "en":
        style_instructions = {
            "analysis": "Provide a sharp analysis or insightful opinion on why this moment is important or interesting.",
            "reaction": "Provide a natural reaction as if you are watching this moment and are impressed/surprised.",
            "lesson": "Extract one key lesson or takeaway that the audience can apply from this moment.",
            "summary": "Provide a catchy but brief context or summary of what's happening in this moment."
        }
        
    chosen_style = style_instructions.get(style, style_instructions["analysis"])

    length_instruction = "berdurasi pendek (3-5 kalimat, sekitar 20-40 detik saat diucapkan)"
    if length == "short":
        length_instruction = "berdurasi sangat pendek (1-2 kalimat, sekitar 5-15 detik saat diucapkan)"
    elif length == "long":
        length_instruction = "berdurasi lumayan panjang (5-7 kalimat, sekitar 40-60 detik saat diucapkan)"
    if language == "en":
        length_instruction = "short duration (3-5 sentences, around 20-40 seconds when spoken)"
        if length == "short":
            length_instruction = "very short duration (1-2 sentences, around 5-15 seconds when spoken)"
        elif length == "long":
            length_instruction = "medium duration (5-7 sentences, around 40-60 seconds when spoken)"

    prompt = f"""Kamu adalah seorang narator/komentator video pendek (Shorts/TikTok/Reels).
Tugasmu adalah membuat script voice-over {length_instruction} 
berdasarkan transkrip video berikut.

{lang_instruction}
{chosen_style}

ATURAN:
1. ADD INFORMATION RULE: Sebelum menulis commentary, tanyakan dalam hati: "Apakah ini menambah informasi, interpretasi, konteks, klarifikasi, koneksi, atau setup yang tidak sudah jelas dari source di sekitarnya?"
2. JANGAN sekadar mengulang isi transkrip. Jika commentary hanya meniru arti source yang baru saja diputar, hapus commentary itu. Do not merely restate the immediately preceding transcript.
3. JANGAN menggunakan sapaan pembuka seperti "Halo guys" atau penutup seperti "Jangan lupa subscribe". Langsung to the point ke isi momen.
4. JANGAN berikan elemen format atau penjelasan tambahan. HANYA KELUARKAN TEKS SCRIPT YANG AKAN DIBACAKAN.
5. Commentary must be grounded and supported by the available source. Jangan mengarang statistik, tanggal, fakta, motif, hubungan, atau kutipan yang tidak ada di source.
6. Bila momen itu sudah jelas dari source, lebih baik tidak usah komentar; satu komentar yang bermanfaat lebih baik daripada banyak komentar yang mengulang source.
7. Gunakan gaya bicara natural untuk audio: singkat, konkret, dan terdengar seperti editor yang berpikir, bukan AI yang menulis esai.
8. Hindari filler seperti "ini nggak pernah terjadi" atau "ini sama sekali gila" kecuali memang benar-benar dibenarkan oleh source.

TRANSKRIP KLIP:
\"\"\"
{transcript_snippet}
\"\"\"

SCRIPT VOICE-OVER (Hanya teks yang dibacakan, tanpa tanda kutip di awal/akhir):"""
    return prompt


def get_commentary_timeline_prompt(transcript_snippet: str, style: str, language: str, length: str, clip_duration: float | None = None) -> str:
    clip_duration_hint = ""
    if clip_duration is not None:
        clip_duration_hint = f"\n- Durasi klip sumber yang tersedia: {float(clip_duration):.1f} detik."

    language_rules = "Gunakan bahasa Indonesia yang gaul tapi profesional."
    if language == "en":
        language_rules = "Use engaging, conversational English."

    style_rules = {
        "analysis": "Berikan analisis yang menjelaskan mengapa momen ini penting.",
        "reaction": "Berikan reaksi yang terasa natural dan tidak berlebihan.",
        "lesson": "Titik fokusnya adalah pelajaran dari momen ini.",
        "summary": "Titik fokusnya adalah ringkasan yang memberi context, bukan hanya ulangi transkrip.",
    }.get(style, "Berikan satu analisis yang memperkaya momen ini.")

    if language == "en":
        style_rules = {
            "analysis": "Give analysis that explains why this moment matters.",
            "reaction": "Keep the reaction natural and not repetitive.",
            "lesson": "Focus on the lesson or takeaway from this moment.",
            "summary": "Provide context and a concise summary rather than repeating the transcript.",
        }.get(style, "Give a useful interpretation of this moment.")

    return f"""Kamu adalah editor video pendek yang merancang alur narasi interleaved.
Tugasmu: buatkan plan editorial untuk voice-over yang masuk di antara bagian-bagian sumber asli.

{language_rules}
{style_rules}

Aturan penting:
- ADD INFORMATION RULE: Jaga source footage tetap dominan; commentary hanya muncul ketika benar-benar menambah informasi, interpretasi, konteks, klarifikasi, atau setup yang tidak sudah jelas dari source di sekitarnya.
- Sebelum menerima setiap commentary, tanyakan: "Apakah ini benar-benar menambah value?" Jika jawabannya adalah "tidak, hanya mengulang source", hapus commentary itu.
- Do not merely restate the immediately preceding transcript; do not paraphrase the source as if commentary were a second copy of the same idea.
- Commentary is grounded only in the available source and supplied metadata; unsupported facts are not allowed.
- Gunakan 0-3 intervensi commentary dalam klip standar.
- Pilih natural insertion points di akhir kalimat, jeda, atau transisi topik.
- Tidak boleh ada commentary yang memotong pertanyaan/kalimat penting di tengah.
- Jika tidak ada momen yang layak ditambahkan, bisa return {{"segments": [{{"type": "source", "source_start": 0.0, "source_end": clip_duration}}] }}.
- Commentary harus grounded dan supported by the available source material. Jangan menebak angka, tanggal, motif, hubungan, identitas, fakta, atau kesimpulan yang tidak ada di transkrip.
- Jika ada payoff yang kuat, simpan sebagai tipe 'payoff' dari sumber asli dan hindari menimpa bagian tersebut dengan narasi; protect the payoff.
- Gunakan alasan yang didukung source: context, clarification, analysis, connection, observation, setup, lesson, atau summary yang jelas relevannya.
- Hindari filler generik seperti "here's the crazy part" atau "this changes everything" bila tidak benar-benar masuk akal dari momen yang ada.
- Output HARUS berupa JSON murni, bukan markdown, tanpa komentar tambahan.
- Setiap commentary segment boleh punya field `visual` untuk menentukan gaya visual terbaik di saat narasi sedang dibacakan.
- Visual mode yang diterima: `continue`, `replay`, `zoom`, `broll`.
- Prioritas default: `continue` > `replay` > `zoom` > `broll` > static emergency.
- `continue` adalah mode paling aman dan paling umum; gunakan ini ketika sumber masih relevan atau visual tidak perlu berubah.
- `replay` hanya untuk momen baru-baru ini yang harus ditonton ulang; wajib punya `source_start` dan `source_end` yang valid.
- `zoom` hanya untuk detail seperti ekspresi, gesture, atau objek; gunakan tingkat zoom kecil dan aman (`1.0` hingga `1.15`).
- `broll` hanya jika narasi membahas konsep, tempat, atau objek yang tidak terlihat jelas di source; wajib punya `query` kontekstual yang spesifik.
- B-roll adalah pilihan terakhir setelah CONTINUE, REPLAY, dan ZOOM. Gunakan hanya jika visual konkret benar-benar membantu memahami narasi.
- B-roll memerlukan `query` pencarian 3-8 kata yang spesifik dan `visual_intent` singkat yang menjelaskan fungsi ilustrasinya.
- Jangan gunakan query generik seperti `business`, `money`, atau `technology`; jangan mencari orang/kejadian seolah-olah stok tersebut adalah subjek asli. Hindari visual yang menyiratkan fakta baru.
- Satu visual commentary dipakai untuk seluruh blok VO saat ini. Jika ada beberapa commentary dengan kebutuhan visual berbeda, jangan pilih satu B-roll untuk semuanya; gunakan source-based fallback.
- Jika visual tidak valid, jangan gagal; pakai `continue` secara otomatis.
- Schema target:
{{
  "segments": [
    {{"type": "source", "source_start": 0.0, "source_end": 8.0}},
    {{"type": "commentary", "insert_after_source_time": 8.0, "narration": "This detail matters because...", "visual": {{"mode": "continue"}}}},
    {{"type": "source", "source_start": 8.0, "source_end": 16.0}},
    {{"type": "commentary", "insert_after_source_time": 16.0, "narration": "Now watch how the response changes.", "visual": {{"mode": "replay", "source_start": 12.0, "source_end": 15.5}}}},
    {{"type": "commentary", "insert_after_source_time": 20.0, "narration": "The wafer is produced in a controlled room.", "visual": {{"mode": "broll", "query": "semiconductor clean room manufacturing", "visual_intent": "Illustrate the chip manufacturing environment"}}}}
  ]
}}
- Gunakan angka float untuk waktu.
- source_start dan source_end harus berurutan, tidak overlap, dan tetap dalam durasi klip sumber.
- commentary hanya boleh muncul setelah source yang relevan, bukan di awal tanpa konteks.
- commentary harus mengandung fakta yang didukung transkrip, bukan asumsi luar.
- Keputusan commentary harus punya alasan editorial: context, clarification, analysis, connection, reaction, setup, lesson, atau summary.
- Setelah memilih mode visual, jangan pernah mengirim raw FFmpeg; cukup berikan struktur data yang aman dan valid.

Transkrip klip:
\"\"\"
{transcript_snippet}
\"\"\"
{clip_duration_hint}

Kembalikan JSON valid saja."""


def generate_commentary_plan(transcript_snippet: str, cfg, style="analysis", language="id", length="short", clip_duration: float | None = None) -> dict:
    """Generate a structured commentary/editing plan for the clip from Gemini."""
    snippet = (transcript_snippet or "").strip()
    if not snippet:
        return {"valid": False, "segments": [], "errors": ["Transcript is empty."]}

    api_key = cfg.api_key_gemini
    if not api_key:
        raise ValueError("GOOGLE_API_KEY tidak ditemukan di environment atau config.")

    client = genai.Client(api_key=api_key)
    prompt = get_commentary_timeline_prompt(snippet, style, language, length, clip_duration)

    gemini_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.9,
    )

    model = getattr(cfg, "gemini_model", "gemini-3-flash-preview")
    fallback = getattr(cfg, "gemini_fallback_model", "gemini-2.5-flash")

    raw_text = ""
    for candidate_model in (model, fallback):
        try:
            response = client.models.generate_content(model=candidate_model, contents=prompt, config=gemini_config)
            raw_text = (getattr(response, "text", "") or "").strip()
            if raw_text:
                break
        except Exception:
            continue

    if not raw_text:
        return {"valid": False, "segments": [], "errors": ["Gemini commentary plan generation failed."]}

    try:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            raw_text = raw_text[start:end+1]
        data = json.loads(raw_text)
    except Exception:
        return {"valid": False, "segments": [], "errors": ["Gemini returned malformed JSON."]}

    validation = validate_commentary_plan(data, clip_duration=_safe_float(clip_duration, 0.0))
    if validation["valid"]:
        return {**data, "valid": True, "errors": []}

    safe_clip_duration = _safe_float(clip_duration, 0.0)
    if safe_clip_duration <= 0:
        safe_clip_duration = max(10.0, (len(re.findall(r"\S+", snippet)) * 0.42))

    fallback_narration = "This moment matters because the key tension is building right here, and that makes the decision or response that follows even more important."
    fallback_plan = {
        "segments": [
            {"type": "source", "source_start": 0.0, "source_end": safe_clip_duration * 0.7},
            {"type": "commentary", "insert_after_source_time": safe_clip_duration * 0.7, "narration": fallback_narration},
            {"type": "source", "source_start": safe_clip_duration * 0.7, "source_end": safe_clip_duration},
        ],
        "valid": True,
        "errors": validation["errors"],
    }

    return fallback_plan


def generate_commentary_script(transcript_snippet: str, cfg, style="analysis", language="id", length="short") -> str:
    """Generate commentary script using Gemini AI."""
    print(f"   🧠 Generating {style} commentary script via Gemini ({language}, {length})...")

    plan = generate_commentary_plan(transcript_snippet, cfg, style=style, language=language, length=length)
    narration = _extract_narration_from_plan(plan)
    if narration:
        print(f"   ✅ Structured commentary plan generated ({len(narration)} chars)")
        return narration

    api_key = cfg.api_key_gemini
    if not api_key:
        raise ValueError("GOOGLE_API_KEY tidak ditemukan di environment atau config.")

    client = genai.Client(api_key=api_key)
    prompt = get_commentary_prompt(transcript_snippet, style, language, length)

    gemini_config = types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.9,
    )

    model = getattr(cfg, "gemini_model", "gemini-3-flash-preview")
    fallback = getattr(cfg, "gemini_fallback_model", "gemini-2.5-flash")

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=gemini_config
        )
        if response.text:
            script = response.text.strip().strip('"').strip()
            print(f"   ✅ Script generated ({len(script)} chars)")
            return script
    except Exception as e:
        print(f"   ⚠️ Gemini main model failed: {e}. Trying fallback...")
        try:
            response = client.models.generate_content(
                model=fallback,
                contents=prompt,
                config=gemini_config
            )
            if response.text:
                script = response.text.strip().strip('"').strip()
                print(f"   ✅ Script generated via fallback ({len(script)} chars)")
                return script
        except Exception as e2:
            print(f"   ❌ Gemini fallback failed: {e2}")

    return ""
