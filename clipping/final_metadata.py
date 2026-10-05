"""Final-edit metadata context, validation, and deterministic fallbacks."""

import json
import math
import os
import re
import unicodedata

_GENERIC_TITLES = {
    "you won't believe what happened next",
    "this changes everything",
    "nobody expected this",
    "this is insane",
    "watch until the end",
    "the shocking truth",
}


def clean_text(value):
    return " ".join(str(value or "").split()).strip()


def sanitize_filename(text, fallback="clip"):
    value = unicodedata.normalize("NFKD", clean_text(text)).encode("ascii", "ignore").decode()
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return value[:100] or fallback


def _supported_title(title):
    normalized = clean_text(title)
    lowered = normalized.lower().strip(".!?")
    if not normalized or lowered in _GENERIC_TITLES:
        return False
    if len(normalized) > 100 or len(normalized.split()) > 18:
        return False
    if any(marker in normalized for marker in ("C:\\", "/tmp/", "ffmpeg ")):
        return False
    return True


def choose_title(item):
    """Choose an existing grounded title, with a deterministic non-clickbait fallback."""
    for candidate in (
        item.get("title_inggris"),
        item.get("youtube_title_final"),
        item.get("title_indonesia"),
    ):
        if _supported_title(candidate):
            return clean_text(candidate)

    hook = clean_text(item.get("description_hook"))
    if hook:
        words = hook.split()[:12]
        return " ".join(words).rstrip(".,!? ")
    return "A Specific Moment Worth Understanding"


def choose_thumbnail_text(item):
    """Return concise grounded text distinct from the full title where possible."""
    candidates = [
        item.get("thumbnail_text"),
        item.get("hook_plan", {}).get("text") if isinstance(item.get("hook_plan"), dict) else "",
        item.get("title_inggris"),
    ]
    for candidate in candidates:
        text = clean_text(candidate)
        if not text:
            continue
        words = text.split()[:6]
        result = " ".join(words).upper().strip(".!?")
        if result and result.lower() not in _GENERIC_TITLES:
            return result
    return "THE KEY MOMENT"


def build_final_edit_context(item):
    """Build compact metadata context from the selected/final edit structures."""
    voiceover = item.get("voiceover") if isinstance(item.get("voiceover"), dict) else {}
    plan = voiceover.get("plan") if isinstance(voiceover.get("plan"), dict) else {}
    commentary = []
    for segment in plan.get("segments", []):
        if isinstance(segment, dict) and str(segment.get("type", "")).lower() == "commentary":
            text = clean_text(segment.get("narration") or segment.get("text"))
            if text:
                commentary.append(text)

    return {
        "source_excerpt": clean_text(item.get("source_excerpt") or item.get("alasan")),
        "hook": item.get("hook_plan", {}),
        "commentary": commentary,
        "payoff": item.get("payoff") or item.get("description_context", ""),
        "clip_start": item.get("start_time"),
        "clip_end": item.get("end_time"),
        "pacing_removed_duration": item.get("pacing_removed_duration", 0.0),
        "broll_assets": item.get("broll_assets", []),
        "quality_status": item.get("quality_status") or item.get("status"),
    }


def validate_metadata_package(item):
    """Return technical/content-safety warnings without judging predicted performance."""
    title = choose_title(item)
    thumbnail_text = choose_thumbnail_text(item)
    description = clean_text(
        item.get("youtube_description_final")
        or item.get("description_context")
        or item.get("description_hook")
    )
    hashtags = item.get("hastag") or ""
    warnings = []
    if not _supported_title(title):
        warnings.append("title is empty, too long, or generic")
    if len(thumbnail_text.split()) > 6:
        warnings.append("thumbnail text exceeds six words")
    if not description:
        warnings.append("description is empty")
    if len(clean_text(hashtags).split()) > 5:
        warnings.append("too many hashtags")
    return {
        "title": title,
        "thumbnail_text": thumbnail_text,
        "description": description,
        "warnings": warnings,
        "valid": not warnings,
    }


def write_metadata_sidecar(item, output_path):
    package = validate_metadata_package(item)
    payload = {
        "title": package["title"],
        "thumbnail_text": package["thumbnail_text"],
        "description": package["description"],
        "hashtags": clean_text(item.get("hastag")).split(),
        "keyword_tags": item.get("youtube_tags_final", item.get("keyword_tags", [])),
        "thumbnail_source_time": item.get("thumbnail_source_time"),
        "final_edit_context": build_final_edit_context(item),
        "warnings": package["warnings"],
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload
