"""
clipping.metadata — QA Metadata Preview & Normalization

Maps to the QA Metadata Preview cell in the notebook.
"""

import json
import math


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _normalize_spaces(text):
    return " ".join(str(text or "").split()).strip()


HOOK_PLAN_TYPES = {"source_teaser", "commentary_hook", "text_hook", "question_hook"}


def normalize_hook_plan(item):
    """Validate structured hook intent and fall back to the legacy source teaser."""
    clip_start = _finite_float(item.get("start_time"), 0.0)
    clip_end = _finite_float(item.get("end_time"), clip_start)
    legacy_start = _finite_float(item.get("hook_start_time"), clip_start)
    legacy_end = _finite_float(item.get("hook_end_time"), legacy_start + 3.0)

    if not (clip_end > clip_start):
        clip_end = clip_start + 3.0
    fallback_start = min(max(legacy_start, clip_start), clip_end)
    fallback_end = min(max(legacy_end, fallback_start), clip_end)
    if fallback_end <= fallback_start:
        fallback_start = clip_start
        fallback_end = min(clip_start + 3.0, clip_end)
    if fallback_end <= fallback_start:
        fallback_end = fallback_start + 0.1

    fallback = {
        "type": "source_teaser",
        "duration_target": fallback_end - fallback_start,
        "text": "",
        "source_start": fallback_start,
        "source_end": fallback_end,
        "fallback": True,
    }
    plan = item.get("hook_plan")
    if not isinstance(plan, dict):
        return fallback

    hook_type = str(plan.get("type") or "").strip().lower()
    text = _normalize_spaces(plan.get("text"))
    start = _finite_float(plan.get("source_start"), None)
    end = _finite_float(plan.get("source_end"), None)
    duration = _finite_float(plan.get("duration_target"), None)
    if (
        hook_type not in HOOK_PLAN_TYPES
        or duration is None
        or duration < 0
        or (duration == 0 and hook_type != "source_teaser")
        or start is None
        or end is None
        or end <= start
        or start < clip_start
        or end > clip_end
        or (hook_type != "source_teaser" and not text)
    ):
        return fallback

    if hook_type in {"text_hook", "question_hook"} and len(text.split()) > 12:
        return fallback
    if hook_type == "commentary_hook" and len(text.split()) > 24:
        return fallback

    return {
        "type": hook_type,
        "duration_target": duration,
        "text": text,
        "source_start": start,
        "source_end": end,
        "fallback": False,
    }


def _finite_float(value, default):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def validate_story_candidates(
    candidates,
    source_duration=None,
    min_duration=20.0,
    max_duration=179.0,
    overlap_threshold=0.7,
):
    """Validate source/core timing and suppress near-identical overlapping candidates."""
    if not isinstance(candidates, list):
        return []

    safe_source_duration = _finite_float(source_duration, None)
    validated = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        start_value = next(
            (candidate.get(key) for key in ("start_time", "timing_klip_start", "clip_start", "start") if candidate.get(key) is not None),
            None,
        )
        end_value = next(
            (candidate.get(key) for key in ("end_time", "timing_klip_end", "clip_end", "end") if candidate.get(key) is not None),
            None,
        )
        start = _finite_float(start_value, None)
        end = _finite_float(end_value, None)
        if start is None or end is None or start < 0 or end <= start:
            continue
        duration = end - start
        if duration < min_duration or duration > max_duration:
            continue
        if safe_source_duration is not None and safe_source_duration > 0 and end > safe_source_duration:
            continue

        core_start = _finite_float(candidate.get("core_start_time"), start)
        core_end = _finite_float(candidate.get("core_end_time"), end)
        if core_start is None or core_end is None or core_start < start or core_end > end or core_end <= core_start:
            core_start, core_end = start, end

        raw_structure = candidate.get("story_structure")
        raw_structure = raw_structure if isinstance(raw_structure, dict) else {}
        story_structure = {}
        for key in ("setup", "development", "payoff"):
            value = raw_structure.get(key, False)
            story_structure[key] = value is True or (
                isinstance(value, str) and value.strip().lower() in {"true", "yes", "1"}
            ) or (isinstance(value, (int, float)) and not isinstance(value, bool) and value == 1)

        normalized = dict(candidate)
        normalized.update({
            "start_time": start,
            "end_time": end,
            "core_start_time": core_start,
            "core_end_time": core_end,
            "story_structure": story_structure,
            "viral_score": _finite_float(candidate.get("viral_score"), 0.0),
        })
        validated.append(normalized)

    validated.sort(key=lambda item: item["viral_score"], reverse=True)
    distinct = []
    for candidate in validated:
        candidate_duration = candidate["end_time"] - candidate["start_time"]
        candidate_core_duration = candidate["core_end_time"] - candidate["core_start_time"]
        duplicate = False
        for accepted in distinct:
            overlap = max(
                0.0,
                min(candidate["end_time"], accepted["end_time"])
                - max(candidate["start_time"], accepted["start_time"]),
            )
            shorter_duration = min(candidate_duration, accepted["end_time"] - accepted["start_time"])
            if shorter_duration <= 0 or overlap / shorter_duration < overlap_threshold:
                continue

            core_overlap = max(
                0.0,
                min(candidate["core_end_time"], accepted["core_end_time"])
                - max(candidate["core_start_time"], accepted["core_start_time"]),
            )
            shorter_core = min(
                candidate_core_duration,
                accepted["core_end_time"] - accepted["core_start_time"],
            )
            if shorter_core <= 0 or core_overlap / shorter_core >= 0.5:
                duplicate = True
                break

        if not duplicate:
            distinct.append(candidate)

    return distinct


def _trim_title(text, max_len=100):
    text = _normalize_spaces(text)
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0].strip()
    return cut if cut else text[:max_len].strip()


def _normalize_hashtags(text, min_tags=2, max_tags=3):
    parts = _normalize_spaces(text).split()
    clean = []
    seen = set()

    for p in parts:
        if not p:
            continue
        if not p.startswith("#"):
            p = "#" + p.lstrip("#")
        key = p.lower()
        if key not in seen:
            seen.add(key)
            clean.append(p)
        if len(clean) >= max_tags:
            break

    return " ".join(clean), len(clean)


def _normalize_keyword_tags(tags, min_items=5, max_items=8):
    if not isinstance(tags, list):
        tags = []
    out = []
    seen = set()

    for t in tags:
        x = _normalize_spaces(t)
        if not x:
            continue
        key = x.lower()
        if key not in seen:
            seen.add(key)
            out.append(x)
        if len(out) >= max_items:
            break

    return out


def _build_youtube_description(hook, context, hashtags, source_url=None):
    parts = [
        _normalize_spaces(hook),
        _normalize_spaces(context),
        _normalize_spaces(hashtags),
    ]
    desc = "\n\n".join([p for p in parts if p]).strip()

    # Auto-add source attribution if available
    if source_url:
        desc += f"\n\nSource: {source_url}"

    return desc


def _build_tiktok_caption(caption, hashtags):
    caption = _normalize_spaces(caption)
    hashtags = _normalize_spaces(hashtags)
    if caption and hashtags:
        return f"{caption}\n{hashtags}"
    return caption or hashtags


def _looks_indonesian(text):
    text = f" {_normalize_spaces(text).lower()} "
    indikator = [
        " yang ", " dan ", " untuk ", " dengan ", " karena ", " adalah ",
        " bisa ", " tidak ", " lebih ", " dalam ", " pada ", " agar ",
        " dari ", " ini ", " itu ", " juga ", " kalau ", " saat ",
        " tentang ", " bikin ", " banget ", " jadi ", " sudah ",
    ]
    return any(w in text for w in indikator)


# ==============================================================================
# MAIN API
# ==============================================================================

def normalize_and_validate(
    hasil_json: list[dict],
    source_duration=None,
    min_duration=20.0,
    max_duration=179.0,
) -> list[dict]:
    """
    Normalize and enrich metadata fields, add *_final fields.

    Mutates items in-place and returns sorted list.
    """
    hasil_json = validate_story_candidates(
        hasil_json,
        source_duration=source_duration,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    valid_items = []
    laporan = []
    semua_warning = []
    for item in hasil_json:
        if not isinstance(item, dict):
            print(f"⚠️ Melewati item metadata yang tidak valid (bukan dict): {type(item)}")
            continue

        # 1. FLATTENING: If model put everything in a "metadata" key, bring it up
        if isinstance(item.get("metadata"), dict):
            for k, v in item["metadata"].items():
                if k not in item:
                    item[k] = v

        # 2. ALIASING: Handle variations in field names (especially for non-Gemini providers)
        rank = item.get("rank") or item.get("peringkat") or item.get("no") or "?"
        item["rank"] = rank
        item["viral_score"] = item.get("viral_score", 0)
        
        # Timing aliases
        it_st = item.get("start_time") or item.get("timing_klip_start") or item.get("clip_start") or item.get("start")
        it_en = item.get("end_time") or item.get("timing_klip_end") or item.get("clip_end") or item.get("end")
        item["start_time"] = float(it_st) if it_st is not None else 0.0
        item["end_time"] = float(it_en) if it_en is not None else 0.0

        # Hook aliases (DeepSeek sometimes returns hook as string + hook_start_time at root)
        if isinstance(item.get("hook"), str) and "hook_start_time" in item:
            item["hook"] = {
                "text": item["hook"],
                "start_time": item.get("hook_start_time", item["start_time"]),
                "end_time": item.get("hook_end_time", item["end_time"]),
            }
        
        # Ensure hook exists as dict for later code
        if not isinstance(item.get("hook"), dict):
            item["hook"] = {"text": str(item.get("hook", "")), "start_time": item["start_time"], "end_time": item["start_time"] + 3.0}

        hook_plan = normalize_hook_plan(item)
        item["hook_plan"] = hook_plan
        item["hook_start_time"] = hook_plan["source_start"]
        item["hook_end_time"] = hook_plan["source_end"]

        item["title_indonesia"] = _trim_title(item.get("title_indonesia", ""))
        item["title_inggris"] = _trim_title(item.get("title_inggris", ""))
        item["description_hook"] = _normalize_spaces(item.get("description_hook", ""))
        item["description_context"] = _normalize_spaces(item.get("description_context", ""))
        item["hastag"] = _normalize_spaces(item.get("hastag") or item.get("hashtag") or "")
        item["tiktok_title_id"] = _normalize_spaces(item.get("tiktok_title_id", ""))
        item["tiktok_caption_id"] = _normalize_spaces(item.get("tiktok_caption_id", ""))
        item["tiktok_caption"] = _normalize_spaces(item.get("tiktok_caption", ""))
        item["keyword_tags"] = _normalize_keyword_tags(item.get("keyword_tags", []))

        hastag_clean, hashtag_count = _normalize_hashtags(item["hastag"])
        item["hastag"] = hastag_clean

        # Enriched fields
        item["youtube_title_final"] = item["title_inggris"]
        item["youtube_description_final"] = _build_youtube_description(
            item.get("description_hook", ""),
            item.get("description_context", ""),
            item.get("hastag", ""),
            source_url=item.get("source_url"),
        )
        item["youtube_tags_final"] = item.get("keyword_tags", [])

        # TikTok EN (existing behavior)
        item["tiktok_caption_final"] = _build_tiktok_caption(
            item.get("tiktok_caption", ""),
            item.get("hastag", ""),
        )

        # TikTok ID (new fields)
        item["tiktok_title_id_final"] = item.get("tiktok_title_id", "") or item.get("title_indonesia", "")
        item["tiktok_caption_id_final"] = _build_tiktok_caption(
            item.get("tiktok_caption_id", ""),
            item.get("hastag", ""),
        )

        # --- Warnings ---
        warning = []
        if not item["title_indonesia"]:
            warning.append("title_indonesia kosong")
        if len(item["title_indonesia"]) > 100:
            warning.append("title_indonesia > 100 karakter")

        if not item["title_inggris"]:
            warning.append("title_inggris kosong")
        if len(item["title_inggris"]) > 100:
            warning.append("title_inggris > 100 karakter")

        if hashtag_count < 2 or hashtag_count > 3:
            warning.append("jumlah hashtag bukan 2-3")

        if not item["description_hook"]:
            warning.append("description_hook kosong")
        if not item["description_context"]:
            warning.append("description_context kosong")
        if len(item["keyword_tags"]) < 5:
            warning.append("keyword_tags terlalu sedikit")

        if not item["tiktok_title_id"]:
            warning.append("tiktok_title_id kosong")
        if not item["tiktok_caption_id"]:
            warning.append("tiktok_caption_id kosong")
        if not item["tiktok_caption"]:
            warning.append("tiktok_caption kosong")

        # Safety: warn if description is too short (looks like spam/lazy reupload)
        yt_desc = item.get("youtube_description_final", "")
        if len(yt_desc) < 30:
            warning.append("youtube_description terlalu pendek (< 30 karakter) — terlihat seperti spam")

        if _looks_indonesian(item["title_inggris"]):
            warning.append("title_inggris terdeteksi bukan English penuh")
        if _looks_indonesian(item["description_hook"]):
            warning.append("description_hook terdeteksi bukan English penuh")
        if _looks_indonesian(item["description_context"]):
            warning.append("description_context terdeteksi bukan English penuh")
        if _looks_indonesian(item["tiktok_caption"]):
            warning.append("tiktok_caption terdeteksi bukan English penuh")

        if item["tiktok_title_id"] and not _looks_indonesian(item["tiktok_title_id"]):
            warning.append("tiktok_title_id terdeteksi bukan Bahasa Indonesia")
        if item["tiktok_caption_id"] and not _looks_indonesian(item["tiktok_caption_id"]):
            warning.append("tiktok_caption_id terdeteksi bukan Bahasa Indonesia")

        if warning:
            semua_warning.append((rank, warning))
            
        item["_warnings_temp"] = " | ".join(warning) if warning else "OK"
        valid_items.append(item)

    # Sort based on viral_score (descending)
    valid_items = sorted(valid_items, key=lambda x: x.get("viral_score", 0), reverse=True)
    
    # Re-assign rank to be purely sequential and build laporan
    for idx, item in enumerate(valid_items):
        item["rank"] = idx + 1
        klasifikasi = item.get("klasifikasi_akun", {})
        
        laporan.append({
            "rank": item["rank"],
            "viral_score": item.get("viral_score", 0),
            "durasi": round(float(item.get("end_time", 0)) - float(item.get("start_time", 0)), 2),
            "akun_tujuan": klasifikasi.get("akun_tujuan", ""),
            "title_indonesia": item["title_indonesia"],
            "title_inggris": item["title_inggris"],
            "tiktok_title_id": item["tiktok_title_id"],
            "hashtags": item["hastag"],
            "warnings": item.pop("_warnings_temp", "OK"),
        })

    return valid_items


def print_preview(hasil_json: list[dict]) -> None:
    """Print a human-readable metadata preview to stdout."""
    print("✅ Preview metadata siap.")
    print("Field tambahan yang dibuat:")
    print("- youtube_title_final")
    print("- youtube_description_final")
    print("- youtube_tags_final")
    print("- tiktok_caption_final")
    print("- tiktok_title_id_final")
    print("- tiktok_caption_id_final")
    print()

    print("===== PREVIEW DETAIL PER KLIP =====")
    for item in hasil_json:
        klasifikasi = item.get("klasifikasi_akun", {})
        print(f"\n--- Rank {item['rank']} (Viral Score: {item.get('viral_score', '?')}) ---")
        print(f"Akun Tujuan       : {klasifikasi.get('akun_tujuan', '')} ({klasifikasi.get('tipe_akun', '')})")
        print(f"Alasan Akun       : {klasifikasi.get('alasan', '')}")
        print(f"Title ID          : {item['title_indonesia']}")
        print(f"Title EN          : {item['title_inggris']}")
        print(f"TikTok Title ID   : {item.get('tiktok_title_id_final', '')}")
        print(f"Hashtag           : {item['hastag']}")
        print(f"Hook Desc         : {item['description_hook']}")
        print(f"Ctx Desc          : {item['description_context']}")
        print(f"YT Desc           : {item['youtube_description_final']}")
        print(f"YT Tags           : {item['youtube_tags_final']}")
        print(f"TikTok EN         : {item['tiktok_caption_final']}")
        print(f"TikTok Caption ID : {item.get('tiktok_caption_id_final', '')}")


def save_metadata_preview(hasil_json: list[dict], path: str = "metadata_preview.json") -> None:
    """Save normalized metadata to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hasil_json, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Disimpan ke {path}")