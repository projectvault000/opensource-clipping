"""Deterministic subtitle phrase grouping and ASS-safe text helpers."""

import math
import re
import string


_SOFT_BREAK_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "if", "in",
    "is", "it", "of", "on", "or", "so", "that", "the", "to", "was", "were",
    "when", "which", "who", "with", "you", "your",
}
_STRONG_END = ".?!:;…"


def _finite(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clean_word(word):
    return " ".join(str(word or "").split()).strip()


def _word_key(word):
    return _clean_word(word).lower().strip(string.punctuation + "…")


def _is_boundary_word(word):
    return _clean_word(word).rstrip().endswith(tuple(_STRONG_END))


def _normalize_words(words):
    normalized = []
    for item in words if isinstance(words, list) else []:
        if not isinstance(item, dict):
            continue
        text = _clean_word(item.get("word") or item.get("text"))
        start = _finite(item.get("start"))
        end = _finite(item.get("end"))
        if not text or start is None or end is None or end <= start:
            continue
        if normalized and text.lower() == normalized[-1]["word"].lower() and start <= normalized[-1]["end"] + 0.03:
            continue
        normalized.append({**item, "word": text, "start": start, "end": end})
    return normalized


def group_timed_words(words, max_words=5, min_words=2, pause_threshold=0.45):
    """Group timed words into readable phrases without losing original timings."""
    normalized = _normalize_words(words)
    if not normalized:
        return []

    max_words = max(1, int(max_words or 1))
    min_words = max(1, min(int(min_words or 1), max_words))
    pause_threshold = max(_finite(pause_threshold, 0.45), 0.05)
    groups = []
    current = []

    def emit():
        if not current:
            return
        groups.append({
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "text": " ".join(item["word"] for item in current),
            "words": list(current),
        })
        current.clear()

    for index, word in enumerate(normalized):
        current.append(word)
        next_word = normalized[index + 1] if index + 1 < len(normalized) else None
        if next_word is None:
            emit()
            continue

        gap = max(next_word["start"] - word["end"], 0.0)
        punctuation_boundary = _is_boundary_word(word["word"])
        pause_boundary = gap >= pause_threshold
        last_key = _word_key(word["word"])
        at_soft_limit = len(current) >= max_words
        force_limit = len(current) >= max_words + 1

        if force_limit:
            emit()
        elif len(current) >= min_words and (punctuation_boundary or pause_boundary):
            emit()
        elif at_soft_limit and last_key not in _SOFT_BREAK_WORDS:
            emit()

    if current:
        emit()
    return groups


def escape_ass_text(text):
    """Escape transcript text so it cannot create ASS override tags or broken lines."""
    value = str(text or "")
    value = value.replace("\r", " ").replace("\n", "\\N")
    value = value.replace("{", "\\{").replace("}", "\\}")
    return value


def wrap_phrase_words(words, max_line_words=0):
    """Return lightweight balanced line groups; ASS still controls final pixel wrapping."""
    items = [_clean_word(item.get("word") if isinstance(item, dict) else item) for item in words]
    items = [item for item in items if item]
    if not max_line_words or len(items) <= max_line_words:
        return [items]
    midpoint = max(1, min(len(items) - 1, round(len(items) / 2)))
    return [items[:midpoint], items[midpoint:]]


def validate_caption_segments(segments, duration=None):
    """Validate timed caption events without judging their creative quality."""
    errors = []
    safe_duration = _finite(duration, None)
    if not isinstance(segments, list):
        return {"valid": False, "errors": ["Caption segments must be a list."]}

    previous_end = 0.0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            errors.append(f"Caption {index} is not an object.")
            continue
        start = _finite(segment.get("start"), None)
        end = _finite(segment.get("end"), None)
        text = _clean_word(segment.get("text"))
        if start is None or end is None or start < 0 or end <= start:
            errors.append(f"Caption {index} has invalid timing.")
            continue
        if safe_duration is not None and end > safe_duration + 0.05:
            errors.append(f"Caption {index} exceeds the expected duration.")
        if not text:
            errors.append(f"Caption {index} has no text.")
        if start < previous_end - 0.05:
            errors.append(f"Caption {index} overlaps the previous caption.")
        previous_end = max(previous_end, end)
    return {"valid": not errors, "errors": errors, "count": len(segments)}
