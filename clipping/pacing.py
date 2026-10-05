"""Conservative pause shortening based on existing transcript word times."""

import math


LEADING_THRESHOLD = 1.25
TRAILING_THRESHOLD = 1.5
SENTENCE_THRESHOLD = 1.8
CONTINUATION_THRESHOLD = 2.3
HESITATION_THRESHOLD = 2.6
QUESTION_THRESHOLD = 3.2
ELLIPSIS_THRESHOLD = 3.5
LEADING_RETAINED = 0.3
TRAILING_RETAINED = 0.4
SENTENCE_RETAINED = 0.85
CONTINUATION_RETAINED = 1.1
HESITATION_RETAINED = 1.2
DRAMATIC_RETAINED = 1.5
MAX_EDITS_PER_RANGE = 4
MIN_TIME_BETWEEN_EDITS = 2.5


def _finite_float(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _transcript_words(data_segments):
    words = []
    for segment in data_segments if isinstance(data_segments, list) else []:
        if not isinstance(segment, dict):
            continue
        segment_words = segment.get("words")
        if isinstance(segment_words, list) and segment_words:
            candidates = segment_words
        else:
            candidates = [segment]
        for word in candidates:
            if not isinstance(word, dict):
                continue
            start = _finite_float(word.get("start"))
            end = _finite_float(word.get("end"))
            text = str(word.get("word") or word.get("text") or "").strip()
            if start is None or end is None or end <= start or not text:
                continue
            words.append({"start": start, "end": end, "text": text})
    return sorted(words, key=lambda word: (word["start"], word["end"]))


def _base_ranges(source_start, source_end, base_segments, enabled):
    if not enabled or not isinstance(base_segments, list) or len(base_segments) <= 1:
        return [(source_start, source_end)]

    ranges = []
    for segment in base_segments:
        if not isinstance(segment, dict):
            continue
        start = _finite_float(segment.get("start_time"))
        end = _finite_float(segment.get("end_time"))
        if start is None or end is None or end <= start:
            continue
        start = max(source_start, start)
        end = min(source_end, end)
        if end > start:
            ranges.append((start, end))

    if not ranges:
        return [(source_start, source_end)]

    ranges.sort()
    merged = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 0.03:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _pause_policy(previous_text, gap):
    text = previous_text.rstrip()
    if text.endswith(("...", "…")):
        return (ELLIPSIS_THRESHOLD, DRAMATIC_RETAINED, "dramatic")
    if text.endswith(("?", "!")):
        return (QUESTION_THRESHOLD, DRAMATIC_RETAINED, "dramatic")
    if text.endswith((".", ";", ":")):
        return (SENTENCE_THRESHOLD, SENTENCE_RETAINED, "sentence")
    if text.endswith(","):
        return (CONTINUATION_THRESHOLD, CONTINUATION_RETAINED, "continuation")
    return (HESITATION_THRESHOLD, HESITATION_RETAINED, "hesitation")


def build_pacing_edit_map(
    source_start,
    source_end,
    data_segments,
    base_segments=None,
    enabled=True,
):
    """Build synchronized retained source spans and a source-to-output time map.

    Only long transcript gaps are shortened. Video, source audio, and source
    subtitles can all be rendered from the returned ``render_segments``.
    """
    safe_start = _finite_float(source_start)
    safe_end = _finite_float(source_end)
    if safe_start is None or safe_end is None or safe_start < 0 or safe_end <= safe_start:
        return {"render_segments": [], "map": [], "compressed_pauses": [], "changed": False,
                "source_duration": 0.0, "output_duration": 0.0, "removed_duration": 0.0,
                "candidate_count": 0, "shortened_pause_count": 0, "preserved_pause_count": 0}

    ranges = _base_ranges(safe_start, safe_end, base_segments, enabled)
    words = _transcript_words(data_segments)
    cuts = []
    pauses = []
    candidate_count = 0
    preserved_pause_count = 0

    if enabled and words:
        for range_start, range_end in ranges:
            local_words = [
                word for word in words
                if word["end"] > range_start and word["start"] < range_end
            ]
            if not local_words:
                continue

            range_candidates = []
            first_word = local_words[0]
            leading_gap = first_word["start"] - range_start
            if leading_gap >= LEADING_THRESHOLD:
                candidate_count += 1
                retained = min(LEADING_RETAINED, leading_gap)
                cut_end = first_word["start"] - retained
                if cut_end > range_start:
                    range_candidates.append({
                        "cut_start": range_start,
                        "cut_end": cut_end,
                        "source_start": cut_end,
                        "source_end": first_word["start"],
                        "retained_duration": first_word["start"] - cut_end,
                        "kind": "leading",
                    })

            for previous, following in zip(local_words, local_words[1:]):
                gap_start = max(previous["end"], range_start)
                gap_end = min(following["start"], range_end)
                gap = gap_end - gap_start
                if gap < 0.7:
                    continue
                threshold, retained, kind = _pause_policy(previous["text"], gap)
                if gap < threshold:
                    preserved_pause_count += 1
                    continue
                candidate_count += 1
                retained = min(retained, gap)
                removed = gap - retained
                cut_start = gap_start + retained / 2.0
                cut_end = gap_end - retained / 2.0
                if cut_end > cut_start:
                    range_candidates.append({
                        "cut_start": cut_start,
                        "cut_end": cut_end,
                        "source_start": gap_start,
                        "source_end": gap_end,
                        "retained_duration": retained,
                        "kind": kind,
                    })

            last_word = local_words[-1]
            trailing_gap = range_end - last_word["end"]
            if trailing_gap >= TRAILING_THRESHOLD:
                candidate_count += 1
                retained = min(TRAILING_RETAINED, trailing_gap)
                cut_start = last_word["end"] + retained
                if range_end > cut_start:
                    range_candidates.append({
                        "cut_start": cut_start,
                        "cut_end": range_end,
                        "source_start": last_word["end"],
                        "source_end": cut_start,
                        "retained_duration": cut_start - last_word["end"],
                        "kind": "trailing",
                    })

            range_candidates.sort(key=lambda edit: edit["cut_start"])
            accepted = []
            for edit in range_candidates:
                if len(accepted) >= MAX_EDITS_PER_RANGE:
                    preserved_pause_count += 1
                    continue
                if accepted and edit["cut_start"] - accepted[-1]["cut_end"] < MIN_TIME_BETWEEN_EDITS:
                    preserved_pause_count += 1
                    continue
                edit["removed_duration"] = max(
                    edit["source_end"] - edit["source_start"] - edit["retained_duration"],
                    0.0,
                )
                accepted.append(edit)
            cuts.extend(accepted)
            pauses.extend(accepted)

    render_segments = []
    mapping = []
    output_cursor = 0.0
    for range_start, range_end in ranges:
        local_cuts = [cut for cut in cuts if cut["cut_start"] >= range_start and cut["cut_end"] <= range_end]
        cursor = range_start
        for cut in local_cuts:
            if cut["cut_start"] > cursor:
                render_segments.append({"start_time": cursor, "end_time": cut["cut_start"]})
            cursor = max(cursor, cut["cut_end"])
        if range_end > cursor:
            render_segments.append({"start_time": cursor, "end_time": range_end})

        range_pauses = [pause for pause in pauses if pause["source_start"] >= range_start and pause["source_end"] <= range_end]
        first_source = range_start
        leading = next((p for p in range_pauses if p["kind"] == "leading"), None)
        if leading:
            first_source = leading["source_start"]
        last_source = range_end
        trailing = next((p for p in range_pauses if p["kind"] == "trailing"), None)
        if trailing:
            last_source = trailing["source_end"]

        cursor = first_source
        for pause in sorted(
            (p for p in range_pauses if p["kind"] not in {"leading", "trailing"}),
            key=lambda item: item["source_start"],
        ):
            pause_start = pause["source_start"]
            pause_end = pause["source_end"]
            if pause_start > cursor:
                span = pause_start - cursor
                mapping.append({"source_start": cursor, "source_end": pause_start,
                                "output_start": output_cursor, "output_end": output_cursor + span,
                                "kind": "source"})
                output_cursor += span
            pause_end_output = output_cursor + pause["retained_duration"]
            mapping.append({"source_start": pause_start, "source_end": pause_end,
                            "output_start": output_cursor, "output_end": pause_end_output,
                            "kind": "compressed_pause"})
            pause["output_start"] = output_cursor
            pause["output_end"] = pause_end_output
            output_cursor = pause_end_output
            cursor = pause_end

        if last_source > cursor:
            span = last_source - cursor
            mapping.append({"source_start": cursor, "source_end": last_source,
                            "output_start": output_cursor, "output_end": output_cursor + span,
                            "kind": "source"})
            output_cursor += span

    source_duration = sum(end - start for start, end in ranges)
    output_duration = sum(seg["end_time"] - seg["start_time"] for seg in render_segments)
    removed_duration = max(source_duration - output_duration, 0.0)
    return {
        "render_segments": render_segments,
        "map": mapping,
        "compressed_pauses": pauses,
        "changed": removed_duration > 1e-6,
        "source_duration": source_duration,
        "output_duration": output_duration,
        "removed_duration": removed_duration,
        "candidate_count": candidate_count,
        "shortened_pause_count": len(pauses),
        "preserved_pause_count": preserved_pause_count,
    }


def map_source_time(edit_map, source_time):
    """Map a source timestamp into the paced output timeline, clamping removed gaps."""
    source_time = _finite_float(source_time)
    if source_time is None or not isinstance(edit_map, list) or not edit_map:
        return None

    for entry in edit_map:
        start = _finite_float(entry.get("source_start"))
        end = _finite_float(entry.get("source_end"))
        output_start = _finite_float(entry.get("output_start"))
        output_end = _finite_float(entry.get("output_end"))
        if None in (start, end, output_start, output_end) or end <= start:
            continue
        if start <= source_time <= end:
            fraction = min(max((source_time - start) / (end - start), 0.0), 1.0)
            return output_start + fraction * (output_end - output_start)

    endpoints = []
    for entry in edit_map:
        endpoints.extend((
            (_finite_float(entry.get("source_start")), _finite_float(entry.get("output_start"))),
            (_finite_float(entry.get("source_end")), _finite_float(entry.get("output_end"))),
        ))
    endpoints = [(source, output) for source, output in endpoints if source is not None and output is not None]
    if not endpoints:
        return None
    return min(endpoints, key=lambda point: abs(point[0] - source_time))[1]


def shift_output_map(edit_map, output_offset):
    """Shift a main-source map into the final timeline after hook/VO prefixes."""
    offset = max(_finite_float(output_offset, 0.0), 0.0)
    shifted = []
    for entry in edit_map if isinstance(edit_map, list) else []:
        item = dict(entry)
        item["output_start"] = _finite_float(item.get("output_start"), 0.0) + offset
        item["output_end"] = _finite_float(item.get("output_end"), 0.0) + offset
        shifted.append(item)
    return shifted


def validate_pacing_map(edit_map, expected_duration, tolerance=0.05):
    """Check source/output ranges are finite, ordered, and within expected output."""
    expected = _finite_float(expected_duration)
    if expected is None or expected <= 0 or not isinstance(edit_map, list) or not edit_map:
        return {"valid": False, "errors": ["Pacing map or expected duration is missing/invalid."]}

    errors = []
    previous_source_end = None
    previous_output_end = None
    for index, entry in enumerate(edit_map):
        if not isinstance(entry, dict):
            errors.append(f"Map entry {index} must be an object.")
            continue
        source_start = _finite_float(entry.get("source_start"))
        source_end = _finite_float(entry.get("source_end"))
        output_start = _finite_float(entry.get("output_start"))
        output_end = _finite_float(entry.get("output_end"))
        if None in (source_start, source_end, output_start, output_end):
            errors.append(f"Map entry {index} contains non-finite timing.")
            continue
        if source_start < 0 or source_end <= source_start:
            errors.append(f"Map entry {index} has an invalid source interval.")
        if output_start < 0 or output_end <= output_start:
            errors.append(f"Map entry {index} has an invalid output interval.")
        if previous_source_end is not None and source_start < previous_source_end - tolerance:
            errors.append(f"Map entry {index} overlaps the previous source interval.")
        if previous_output_end is not None and output_start < previous_output_end - tolerance:
            errors.append(f"Map entry {index} overlaps the previous output interval.")
        if output_end > expected + tolerance:
            errors.append(f"Map entry {index} exceeds expected final duration.")
        previous_source_end = source_end
        previous_output_end = output_end

    if previous_output_end is not None and abs(previous_output_end - expected) > max(tolerance, expected * 0.01):
        errors.append("Pacing map end does not match expected final duration.")
    return {"valid": not errors, "errors": errors}
