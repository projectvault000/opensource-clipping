"""
Core Studio pipeline implementation extracted from clipping/studio.py.

This module acts as an orchestrator, importing from modularized subcomponents
to maintain original behavior while allowing clipping/studio.py to remain
a thin orchestration and compatibility entry point.
"""

import html
import importlib.util
import json
import math
import os
import random
import re
import shutil
import string
import subprocess
import textwrap
import time
import urllib.parse
import urllib.request

import cv2
from .. import audio_mastering, broll_policy, final_metadata, pacing, voiceover
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont
from yt_dlp import YoutubeDL
from ..visual_plan import compute_voiceover_visual_plan as _compute_voiceover_visual_plan

def _load_studio_internal_module(file_name: str, module_alias: str):
    module_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import modules to re-export for studio.py
utils = _load_studio_internal_module("utils.py", "clipping_studio_utils")
_get_cv2_interpolation = utils._get_cv2_interpolation
_resize_frame = utils._resize_frame
_get_render_dims = utils._get_render_dims
_is_vertical_ratio = utils._is_vertical_ratio
face_detection = _load_studio_internal_module("face_detection.py", "clipping_studio_face_detection")
get_face_detector = face_detection.get_face_detector
estimate_speaker_count_from_video = face_detection.estimate_speaker_count_from_video
typography = _load_studio_internal_module("typography.py", "clipping_studio_typography")
download_google_font = typography.download_google_font
register_fonts_for_libass = typography.register_fonts_for_libass
siapkan_font_tipografi = typography.siapkan_font_tipografi
audio_bgm = _load_studio_internal_module("audio_bgm.py", "clipping_studio_audio_bgm")
get_local_bgm_file = audio_bgm.get_local_bgm_file
build_bgm_filter = audio_bgm.build_bgm_filter
broll = _load_studio_internal_module("broll.py", "clipping_studio_broll")
download_pexels_broll = broll.download_pexels_broll
crop_center_broll = broll.crop_center_broll
subtitles = _load_studio_internal_module("subtitles.py", "clipping_studio_subtitles")
buat_file_ass = subtitles.buat_file_ass
effects = _load_studio_internal_module("effects.py", "clipping_studio_effects")
siapkan_glitch_video = effects.siapkan_glitch_video
transitions = _load_studio_internal_module("transitions.py", "clipping_studio_transitions")
download_transition_raw = transitions.download_transition_raw
download_all_transitions = transitions.download_all_transitions
get_random_transition = transitions.get_random_transition
prepare_transition_clip = transitions.prepare_transition_clip
TMP_TRANSITION_POOL = transitions.TMP_TRANSITION_POOL
thumbnail = _load_studio_internal_module("thumbnail.py", "clipping_studio_thumbnail")
buat_thumbnail = thumbnail.buat_thumbnail
render_hybrid = _load_studio_internal_module("render_hybrid.py", "clipping_studio_render_hybrid")
buat_video_hybrid = render_hybrid.buat_video_hybrid
render_split_screen = _load_studio_internal_module("render_split_screen.py", "clipping_studio_render_split_screen")
buat_video_split_screen = render_split_screen.buat_video_split_screen
render_camera_switch = _load_studio_internal_module("render_camera_switch.py", "clipping_studio_render_camera_switch")
buat_video_camera_switch = render_camera_switch.buat_video_camera_switch

# Helpers and ffmpeg_utils
_helpers = _load_studio_internal_module("helpers.py", "clipping_studio_helpers")
_ffmpeg_utils = _load_studio_internal_module("ffmpeg_utils.py", "clipping_studio_ffmpeg_utils")

format_seconds = _helpers.format_seconds
escape_ffmpeg_filter_value = _helpers.escape_ffmpeg_filter_value
detect_video_encoder = _ffmpeg_utils.detect_video_encoder
get_ts_encode_args = _ffmpeg_utils.get_ts_encode_args
get_mp4_encode_args = _ffmpeg_utils.get_mp4_encode_args
open_ffmpeg_video_writer = _ffmpeg_utils.open_ffmpeg_video_writer
build_ffmpeg_progress_cmd = _ffmpeg_utils.build_ffmpeg_progress_cmd
run_ffmpeg_with_progress = _ffmpeg_utils.run_ffmpeg_with_progress
v2_helpers = _load_studio_internal_module("v2_helpers.py", "clipping_studio_v2_helpers")
edge_glow_mod = _load_studio_internal_module("edge_glow.py", "clipping_studio_edge_glow")
generate_edge_glow_video = edge_glow_mod.generate_edge_glow_video


def _append_hook_headline(ass_path: str, text: str, duration: float) -> None:
    """Add a brief hook headline in the top safe region of the existing ASS track."""
    safe_text = str(text or "").replace("\\", "").replace("{", "").replace("}", "")
    safe_text = " ".join(safe_text.split())
    safe_duration = max(float(duration or 0.0), 0.1)
    if not safe_text or not os.path.exists(ass_path):
        return
    end_time = f"0:00:{safe_duration:05.2f}"
    with open(ass_path, "a", encoding="utf-8") as ass_file:
        ass_file.write(
            f"Dialogue: 1,0:00:00.00,{end_time},Default,,0,0,0,,{{\\an8\\bord4\\shad1}}{safe_text}\n"
        )


def _build_commentary_visual_source(
    output_path: str,
    source_path: str,
    mode: str,
    source_start: float,
    source_end: float,
    narration_duration: float,
    cfg=None,
    ratio="9:16",
    label="VO visual",
) -> bool:
    """Render commentary visuals through the shared stable vertical crop when available."""
    if not source_path or not os.path.exists(source_path):
        return False

    safe_mode = str(mode or "continue").lower().strip() or "continue"
    safe_duration = max(float(narration_duration or 0.0), 0.5)
    safe_start = max(float(source_start or 0.0), 0.0)
    safe_end = max(float(source_end or safe_start + 0.1), safe_start + 0.1)
    segment_dur = max(safe_end - safe_start, 0.1)

    framed_path = output_path + ".framed.mp4"
    framed_ready = False
    try:
        if cfg is not None:
            try:
                buat_video_hybrid(
                    source_path,
                    framed_path,
                    safe_start,
                    safe_end,
                    ratio,
                    cfg,
                    [],
                    label=label,
                )
                framed_ready = os.path.exists(framed_path) and os.path.getsize(framed_path) > 0
            except Exception as exc:
                print(f"⚠️ [Framing] {label} analysis unavailable; using safe crop fallback ({exc}).")

        vf = "fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2"
        if safe_mode == "zoom":
            vf = "fps=30,scale=iw*1.08:ih*1.08:eval=frame,crop=iw/1.08:ih/1.08,scale=trunc(iw/2)*2:trunc(ih/2)*2"

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        if framed_ready:
            if segment_dur < safe_duration:
                cmd += ["-stream_loop", "-1"]
            cmd += ["-i", framed_path]
        else:
            if segment_dur < safe_duration:
                cmd += ["-stream_loop", "-1"]
            cmd += ["-ss", str(safe_start), "-to", str(safe_end), "-i", source_path]

        cmd += [
            "-vf", vf,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-an",
            "-t", str(safe_duration),
            output_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        return os.path.exists(output_path)
    except subprocess.CalledProcessError:
        return False
    finally:
        if os.path.exists(framed_path):
            os.remove(framed_path)


def proses_klip(
    rank, clip, rasio, glitch_ts, data_segmen, cfg, video_encoder, diarization_data=None
):
    """
    Run full clip processing pipeline from render to final output files.

    Args:
        rank: Clip rank/index.
        clip: Clip metadata object.
        rasio: Target output ratio.
        glitch_ts: Optional prepared glitch transition path.
        data_segmen: Transcript segments.
        cfg: Runtime config object.
        video_encoder: Encoder descriptor dict.
        diarization_data: Optional speaker diarization metadata.

    Returns:
        Manifest dictionary describing processing result and output paths.
    """
    get_x_h = None
    get_x_main = None
    hook_plan = clip.get("hook_plan") if isinstance(clip.get("hook_plan"), dict) else {}
    hook_type = str(hook_plan.get("type") or "source_teaser")
    h_start = float(hook_plan.get("source_start", clip.get("hook_start_time", clip["start_time"])))
    h_end = float(
        hook_plan.get("source_end", clip.get(
            "hook_end_time",
            clip.get("hook_start_time", clip["start_time"]) + cfg.durasi_hook,
        ))
    )
    
    # Custom Hook Override
    file_hook_src = cfg.file_video_asli
    custom_hook = clip.get("custom_hook_info")
    if custom_hook:
        file_hook_src = custom_hook["file_path"]
        h_start = getattr(cfg, "hook_source_start", 0.0)
        
        try:
            cap_h = cv2.VideoCapture(file_hook_src)
            fps = cap_h.get(cv2.CAP_PROP_FPS)
            frames = cap_h.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0:
                vid_duration = frames / fps
            else:
                vid_duration = float('inf')
            cap_h.release()
        except:
            vid_duration = float('inf')

        h_end = h_start + cfg.durasi_hook
        if h_end > vid_duration:
            h_end = vid_duration
    m_start = float(clip["start_time"])
    m_end = float(clip["end_time"])
    judul = clip.get("title_indonesia")
    judul_en = clip.get("title_inggris")

    raw_keep_segments = clip.get("keep_segments")
    pacing_result = pacing.build_pacing_edit_map(
        m_start,
        m_end,
        data_segmen,
        base_segments=raw_keep_segments,
        enabled=not getattr(cfg, "no_segment_trim", False),
    )
    if pacing_result["candidate_count"] or pacing_result["removed_duration"]:
        print(
            f"[Pacing] Detected {pacing_result['candidate_count']} long pause candidates; "
            f"shortened {pacing_result['shortened_pause_count']}, "
            f"preserved {pacing_result['preserved_pause_count']}."
        )
        print(
            f"[Pacing] Retained source {pacing_result['source_duration']:.1f}s "
            f"→ output {pacing_result['output_duration']:.1f}s; "
            f"removed {pacing_result['removed_duration']:.1f}s."
        )

    legacy_transition_duration = (
        1.0
        if not getattr(cfg, "hook_v2", False)
        and getattr(cfg, "use_hook_glitch", True)
        and getattr(cfg, "durasi_hook", 3) > 0
        and glitch_ts
        and os.path.exists(glitch_ts)
        else 0.0
    )
    hook_output_offset = voiceover.estimate_hook_output_offset(
        clip,
        cfg,
        legacy_transition_duration=legacy_transition_duration,
    )
    
    out_vid = os.path.join(cfg.outputs_dir, f"highlight_rank_{rank}_ready.mp4")
    if getattr(cfg, "dev_mode_with_output_merge", False):
        out_vid = os.path.join(cfg.outputs_dir, f"highlight_rank_{rank}_dev_mode_merge_ready.mp4")
        
    out_thm = os.path.join(cfg.outputs_dir, f"thumbnail_rank_{rank}.jpg")

    # Ambil resolusi video asli untuk perhitungan posisi subtitle di dev-mode
    cap_asli = cv2.VideoCapture(cfg.file_video_asli)
    sw = int(cap_asli.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap_asli.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_asli.release()
    source_dim = (sw, sh)

    vo_plan = clip.get("voiceover", {}).get("plan") if isinstance(clip.get("voiceover"), dict) else None
    vo_timeline = voiceover.build_commentary_timeline(
        vo_plan,
        clip_duration=max(float(m_end) - float(m_start), 0.0),
        initial_output_offset=hook_output_offset,
    ) if isinstance(vo_plan, dict) else []
    for timeline_segment in vo_timeline:
        if timeline_segment.get("type") == "commentary":
            source_anchor = m_start + float(timeline_segment.get("insert_after_source_time", 0.0))
            timeline_segment["pacing_main_output_anchor"] = pacing.map_source_time(
                pacing_result["map"],
                source_anchor,
            )

    manifest_item = {
        "rank": rank,
        "status": "pending",
        "ratio": rasio,
        "video_path": out_vid,
        "thumbnail_path": out_thm,
        "thumbnail_text": judul_en or judul or f"Highlight {rank}",
        "youtube_title_final": clip.get(
            "youtube_title_final", clip.get("title_inggris", "")
        ),
        "youtube_description_final": clip.get("youtube_description_final", ""),
        "youtube_tags_final": clip.get("youtube_tags_final", []),
        "tiktok_caption_final": clip.get(
            "tiktok_caption_final", clip.get("hastag", "")
        ),
        "title_indonesia": clip.get("title_indonesia", ""),
        "title_inggris": clip.get("title_inggris", ""),
        "hastag": clip.get("hastag", ""),
        "start_time": m_start,
        "end_time": m_end,
        "hook_start_time": h_start,
        "hook_end_time": h_end,
        "hook_plan": hook_plan,
        "hook_output_offset": hook_output_offset,
        "duration": round(m_end - m_start, 2),
        "pacing_map": pacing_result["map"],
        "pacing_source_duration": pacing_result["source_duration"],
        "pacing_output_duration": pacing_result["output_duration"],
        "pacing_removed_duration": pacing_result["removed_duration"],
        "alasan": clip.get("alasan", ""),
        "broll_list": clip.get("broll_list", []),
        "broll_assets": [],
        "broll_fallbacks": [],
        "audio_mastering": {"status": "pending"},
        "typography_plan": clip.get("typography_plan", []),
        "voiceover_plan": vo_plan,
        "voiceover_timeline": vo_timeline,
    }
    metadata_package = final_metadata.validate_metadata_package(manifest_item)
    manifest_item["youtube_title_final"] = metadata_package["title"]
    manifest_item["title_inggris"] = metadata_package["title"]
    if not manifest_item.get("youtube_description_final"):
        manifest_item["youtube_description_final"] = metadata_package["description"]
    manifest_item["thumbnail_text"] = metadata_package["thumbnail_text"]
    manifest_item["metadata_warnings"] = metadata_package["warnings"]
    manifest_item["final_edit_context"] = final_metadata.build_final_edit_context(manifest_item)

    print(f"\n{'=' * 70}")
    print(f"🔥 [Rank {rank}] Memproses clip")
    print(f"📝 [Judul Indo]   : '{clip.get('title_indonesia', '-')}'")
    print(f"📝 [Judul Inggris]: '{clip.get('title_inggris', '-')}'")
    print(f"#️⃣ [Hastag]      : '{clip.get('hastag', '-')}'")
    print(f"🧠 Encoder aktif  : {video_encoder['name']}")
    print(f"{'=' * 70}")

    typography_plan = clip.get("typography_plan", [])
    siapkan_font_tipografi(cfg)

    h_ts, m_ts, a_hook, a_main = (
        f"h_{rank}.ts",
        f"m_{rank}.ts",
        f"ah_{rank}.ass",
        f"am_{rank}.ass",
    )
    h_silent, m_silent = f"h_silent_{rank}.mp4", f"m_silent_{rank}.mp4"
    
    dev_dual = getattr(cfg, "dev_mode_with_output", False)
    h_ts_dev = f"h_{rank}_dev.ts"
    m_ts_dev = f"m_{rank}_dev.ts"
    
    aktif_hook = bool(cfg.use_hook_glitch and (getattr(cfg, "durasi_hook", 3) > 0))


    # Determine if we should use split-screen mode
    if getattr(cfg, "use_split_screen", False) and _is_vertical_ratio(rasio):
        if cfg.split_trigger == "face":
            use_split = True
        else:
            use_split = (
                diarization_data is not None
                and len(set(s["speaker"] for s in diarization_data)) >= 2
            )
    else:
        use_split = False

    # Camera-switch mode (mutually exclusive: split-screen takes precedence)
    use_camera_switch = (
        not use_split
        and getattr(cfg, "use_camera_switch", False)
        and _is_vertical_ratio(rasio)
        and diarization_data
        and len(set(s["speaker"] for s in diarization_data)) >= 2
    )

    broll_list = clip.get("broll_list", [])
    if not isinstance(broll_list, list):
        broll_list = []
    broll_aktif = []
    if cfg.use_broll and broll_list:
        print(f"   🎥 Mendownload {len(broll_list)} video B-Roll dari Pexels...")
        for i, br in enumerate(broll_list):
            if not isinstance(br, dict):
                manifest_item["broll_fallbacks"].append({"reason": "invalid B-roll plan"})
                continue
            query = broll_policy.normalize_broll_query(br.get("search_query"))
            try:
                broll_start = float(br.get("start_time"))
                broll_end = float(br.get("end_time"))
            except (TypeError, ValueError):
                broll_start, broll_end = -1.0, -1.0
            if (
                query is None
                or not math.isfinite(broll_start)
                or not math.isfinite(broll_end)
                or broll_start < m_start
                or broll_end > m_end
                or broll_end <= broll_start
            ):
                manifest_item["broll_fallbacks"].append({
                    "query": str(br.get("search_query") or ""),
                    "reason": "invalid query or source range",
                })
                continue
            if not any(
                float(segment["end_time"]) > broll_start
                and float(segment["start_time"]) < broll_end
                for segment in pacing_result["render_segments"]
            ):
                manifest_item["broll_fallbacks"].append({
                    "query": query,
                    "reason": "B-roll range is outside the retained source timeline",
                })
                continue

            file_broll = os.path.join(cfg.outputs_dir, f"temp_broll_{rank}_{i}.mp4")
            if download_pexels_broll(
                query,
                rasio,
                file_broll,
                cfg.pexels_api_key,
                minimum_duration=(broll_end - broll_start) + 0.1,
                output_dir=cfg.outputs_dir,
                visual_intent=br.get("visual_intent"),
            ):
                br_copy = dict(br)
                br_copy.update({"filepath": file_broll, "search_query": query})
                broll_aktif.append(br_copy)
                asset_metadata = broll_policy.get_asset_metadata(cfg.outputs_dir, file_broll)
                if asset_metadata:
                    manifest_item["broll_assets"].append(asset_metadata)
            else:
                print(f"   ⚠️ B-roll failed for '{query}'; source footage will remain visible.")
                manifest_item["broll_fallbacks"].append({
                    "query": query,
                    "reason": "Pexels search/download/media validation failed",
                })

    std_p = get_ts_encode_args(video_encoder, fps=30)

    try:
        # HOOK
        hook_v2_data = clip.get("hook_v2", {})
        # Hook V2 is independent of --no-hook (which only disables v1 teaser)
        use_hook_v2 = getattr(cfg, "hook_v2", False)

        if use_hook_v2:
            print("   📸 [Hook V2] Rendering Multi-Hook Intro...")
            h_ts_parts = []
            items = hook_v2_data.get("items", []) if hook_v2_data else []

            # Fallback: if AI didn't provide hook_v2 items, generate from hook timing
            if not items:
                num_items = getattr(cfg, "hook_v2_items", 3)
                hook_total = h_end - h_start
                chunk_dur = max(0.5, hook_total / num_items)
                items = []
                for fi in range(num_items):
                    fi_start = h_start + fi * chunk_dur
                    fi_end = min(fi_start + chunk_dur, h_end)
                    if fi_end <= fi_start:
                        break
                    items.append({"start_time": fi_start, "end_time": fi_end, "text": ""})
                print(f"   ⚠️ [Hook V2] AI tidak memberi items, fallback ke {len(items)} potongan dari hook timing.")

            out_w_v2, out_h_v2 = _get_render_dims(cfg, rasio, source_h=sh)
            flash_dur = getattr(cfg, "white_flash_duration", 0.12)

            for i, item in enumerate(items):
                item_start = float(item["start_time"])
                item_end = float(item["end_time"])
                item_silent = f"h_v2_silent_{rank}_{i}.mp4"
                item_ts = f"h_v2_ts_{rank}_{i}.ts"

                # Render visual (face-tracked crop)
                buat_video_hybrid(
                    file_hook_src, item_silent,
                    item_start, item_end, rasio, cfg,
                    label=f"Rank {rank} HookV2 Item {i}",
                )

                # Build video filter + audio mux
                vf_parts = []
                if cfg.video_sharpen:
                    vf_parts.append("unsharp=5:5:0.5:5:5:0.0")

                cmd_item = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", item_silent,
                    "-ss", str(item_start), "-to", str(item_end),
                    "-i", file_hook_src,
                    "-map", "0:v:0", "-map", "1:a:0",
                ]
                if vf_parts:
                    cmd_item += ["-vf", ",".join(vf_parts)]
                cmd_item += std_p

                cmd_item_full = build_ffmpeg_progress_cmd(cmd_item, item_ts)
                run_ffmpeg_with_progress(
                    cmd_item_full, item_end - item_start,
                    label=f"Rank {rank} HookV2 FFmpeg {i}",
                )
                h_ts_parts.append(item_ts)

                # Transition between items AND after the last item (before main clip)
                trans_mp4 = f"h_v2_trans_{rank}_{i}.mp4"
                trans_ts = f"h_v2_trans_{rank}_{i}.ts"
                trans_type = hook_v2_data.get("transition", {}).get("type", "white_flash") if hook_v2_data else "white_flash"
                if "glitch" in trans_type:
                    v2_helpers.create_glitch_transition(
                        trans_mp4, duration=flash_dur,
                        width=out_w_v2, height=out_h_v2,
                    )
                else:
                    v2_helpers.create_white_flash_transition(
                        trans_mp4, duration=flash_dur,
                        width=out_w_v2, height=out_h_v2,
                    )
                # Convert to .ts for concat (stream copy — already encoded by v2_helpers)
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-i", trans_mp4, "-c", "copy",
                     "-bsf:v", "h264_mp4toannexb",
                     "-f", "mpegts", trans_ts],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                h_ts_parts.append(trans_ts)
                lbl = f"Transisi {i+1}" if i < len(items) - 1 else "Transisi akhir"
                print(f"      ⚡ {lbl} ({trans_type}) berhasil ditambahkan.")

            # Concat all hook v2 pieces into h_ts
            if h_ts_parts:
                concat_str = "concat:" + "|".join(h_ts_parts)
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                     "-i", concat_str, "-c", "copy", h_ts],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )

            # Cleanup temp files
            for tf in h_ts_parts:
                if os.path.exists(tf):
                    os.remove(tf)
            for i in range(len(items)):
                for ext_f in [f"h_v2_silent_{rank}_{i}.mp4", f"h_v2_trans_{rank}_{i}.mp4"]:
                    if os.path.exists(ext_f):
                        os.remove(ext_f)

        elif aktif_hook:
            get_x_h = None
            if use_split:
                print("   📸 [Hook] Split-screen render (Custom Hook diabaikan untuk format ini saat ini atau digabung)...")
                get_x_h = buat_video_split_screen(
                    file_hook_src,
                    h_silent,
                    h_start,
                    h_end,
                    rasio,
                    diarization_data if not custom_hook else None,
                    cfg,
                    label=f"Rank {rank} Hook SplitScreen",
                )
            elif use_camera_switch:
                print("   📸 [Hook] Camera switch render...")
                get_x_h = buat_video_camera_switch(
                    file_hook_src,
                    h_silent,
                    h_start,
                    h_end,
                    rasio,
                    diarization_data if not custom_hook else None,
                    cfg,
                    label=f"Rank {rank} Hook CameraSwitch",
                )
            else:
                print("   📸 [Hook] Hybrid render...")
                get_x_h = buat_video_hybrid(
                    file_hook_src,
                    h_silent,
                    h_start,
                    h_end,
                    rasio,
                    cfg,
                    label=f"Rank {rank} Hook",
                )
            
            aktif_advanced_hook = cfg.use_advanced_text_on_hook
            if not cfg.no_subs and not custom_hook:
                buat_file_ass(
                    data_segmen,
                    h_start,
                    h_end,
                    a_hook,
                    rasio,
                    cfg,
                    typography_plan=typography_plan,
                    gunakan_advanced=aktif_advanced_hook,
                    get_x_func=get_x_h,
                    source_dim=source_dim,
                )

                commentary_hook_audio_ready = (
                    hook_type == "commentary_hook"
                    and isinstance(clip.get("voiceover"), dict)
                    and os.path.exists(clip["voiceover"].get("audio_path", ""))
                )
                if hook_type in {"text_hook", "question_hook"} or (
                    hook_type == "commentary_hook" and not commentary_hook_audio_ready
                ):
                    _append_hook_headline(a_hook, hook_plan.get("text", ""), h_end - h_start)

                print("   🎬 [Hook] FFmpeg burn subtitle + audio...")
                esc_ass_hook = escape_ffmpeg_filter_value(os.path.abspath(a_hook))
                esc_fontsdir = escape_ffmpeg_filter_value(os.path.abspath(cfg.font_dir))
                vf_hook_list = [f"subtitles={esc_ass_hook}:fontsdir={esc_fontsdir}"]
            else:
                print(f"   🎬 [Hook] Skip subtitle rendering {'(Custom Hook)' if custom_hook else ''}...")
                vf_hook_list = []
            
            if cfg.video_sharpen:
                vf_hook_list.append("unsharp=5:5:0.5:5:5:0.0")
            
            vf_hook = ",".join(vf_hook_list) if vf_hook_list else None

            cmd_h_base = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "verbose",
                "-y",
                "-i",
                h_silent,
                "-ss",
                str(h_start),
                "-to",
                str(h_end),
                "-i",
                file_hook_src,
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]
            if vf_hook:
                cmd_h_base += ["-vf", vf_hook]
            cmd_h_base += std_p

            cmd_h = build_ffmpeg_progress_cmd(cmd_h_base, h_ts)
            rc_h, err_h = run_ffmpeg_with_progress(
                cmd_h, h_end - h_start, label=f"Rank {rank} Hook FFmpeg"
            )
            if rc_h != 0:
                raise RuntimeError("FFmpeg hook gagal:\n" + "\n".join(err_h))

        # MAIN
        keep_segments = pacing_result["render_segments"]
        use_segments = (
            keep_segments
            and (
                len(keep_segments) > 1
                or pacing_result["changed"]
                or (isinstance(raw_keep_segments, list) and len(raw_keep_segments) > 1)
            )
            and not getattr(cfg, "no_segment_trim", False)
        )

        if use_segments:
            print(f"   📸 [Main] Rendering {len(keep_segments)} segments (smart trim)...")
            seg_ts_parts = []

            for idx, seg in enumerate(keep_segments):
                s_start = float(seg["start_time"])
                s_end = float(seg["end_time"])
                s_silent = f"m_seg_silent_{rank}_{idx}.mp4"
                s_ass = f"m_seg_ass_{rank}_{idx}.ass"
                s_ts = f"m_seg_ts_{rank}_{idx}.ts"

                # Render visual per segment
                if use_split:
                    get_x_main = buat_video_split_screen(
                        cfg.file_video_asli, s_silent, s_start, s_end,
                        rasio, diarization_data, cfg,
                        label=f"Rank {rank} Seg {idx} SplitScreen",
                        broll_data=broll_aktif,
                    )
                elif use_camera_switch:
                    get_x_main = buat_video_camera_switch(
                        cfg.file_video_asli, s_silent, s_start, s_end,
                        rasio, diarization_data, cfg,
                        label=f"Rank {rank} Seg {idx} CameraSwitch",
                        broll_data=broll_aktif,
                    )
                else:
                    get_x_main = buat_video_hybrid(
                        cfg.file_video_asli, s_silent, s_start, s_end,
                        rasio, cfg, broll_aktif,
                        label=f"Rank {rank} Seg {idx} Hybrid",
                    )

                # Subtitle for this segment
                if not cfg.no_subs:
                    buat_file_ass(
                        data_segmen, s_start, s_end, s_ass, rasio, cfg,
                        typography_plan=typography_plan, gunakan_advanced=True,
                        get_x_func=get_x_main, source_dim=source_dim,
                    )

                esc_ass_seg = escape_ffmpeg_filter_value(os.path.abspath(s_ass)) if not cfg.no_subs else ""
                esc_fontsdir_seg = escape_ffmpeg_filter_value(os.path.abspath(cfg.font_dir))

                vf_seg_parts = [f"subtitles={esc_ass_seg}:fontsdir={esc_fontsdir_seg}"] if not cfg.no_subs else []
                if cfg.video_sharpen:
                    vf_seg_parts.append("unsharp=5:5:0.5:5:5:0.0")

                cmd_s = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", s_silent,
                    "-ss", str(s_start), "-to", str(s_end),
                    "-i", cfg.file_video_asli,
                    "-map", "0:v:0", "-map", "1:a:0",
                ]
                if vf_seg_parts:
                    cmd_s += ["-vf", ",".join(vf_seg_parts)]
                cmd_s += std_p

                cmd_s_full = build_ffmpeg_progress_cmd(cmd_s, s_ts)
                run_ffmpeg_with_progress(
                    cmd_s_full, s_end - s_start,
                    label=f"Rank {rank} Seg FFmpeg {idx}",
                )
                seg_ts_parts.append(s_ts)

            # Concat all segments into m_ts
            concat_str_seg = "concat:" + "|".join(seg_ts_parts)
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-i", concat_str_seg, "-c", "copy", m_ts],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

            # Cleanup segment temps
            for tf in seg_ts_parts:
                if os.path.exists(tf):
                    os.remove(tf)
            for idx in range(len(keep_segments)):
                for ext_f in [f"m_seg_silent_{rank}_{idx}.mp4", f"m_seg_ass_{rank}_{idx}.ass"]:
                    if os.path.exists(ext_f):
                        os.remove(ext_f)

            # Skip the standard MAIN render + subtitle/BGM encoding loop
            # and go directly to BGM application on the concatenated result
            aktif_bgm = cfg.use_auto_bgm
            bgm_mood = clip.get("bgm_mood", "chill")
            if bgm_mood not in getattr(cfg, "bgm_moods", ["chill"]):
                bgm_mood = "chill"
            
            file_bgm = None
            if aktif_bgm:
                print(f"   🎵 Mencari file BGM lokal (Mood: {bgm_mood})...")
                file_bgm = get_local_bgm_file(bgm_mood, getattr(cfg, "bgm_dir", os.path.join(cfg.base_dir, "assets", "bgm")))
                if not file_bgm and bgm_mood != "chill":
                    print("   🔄 Fallback mencari BGM chill...")
                    file_bgm = get_local_bgm_file("chill", getattr(cfg, "bgm_dir", os.path.join(cfg.base_dir, "assets", "bgm")))
                
                if file_bgm:
                    print(f"   ✅ BGM siap: {file_bgm}")
                else:
                    print("   ⚠️ Folder BGM kosong atau file mp3 tidak ditemukan. Render lanjut tanpa BGM.")

            if aktif_bgm and file_bgm:
                print("   🎵 Applying BGM to segmented clip...")
                m_ts_bgm = f"m_bgm_{rank}.ts"
                seg_total_dur = sum(float(s["end_time"]) - float(s["start_time"]) for s in keep_segments)
                bgm_mode = getattr(cfg, "bgm_mode", "ducking")
                filter_complex_seg = build_bgm_filter(
                    bgm_mode, cfg.bgm_base_volume,
                    audio_input_voc="[0:a]", audio_input_bgm="[1:a]"
                )
                cmd_bgm_seg = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", m_ts,
                    "-stream_loop", "-1", "-i", file_bgm,
                    "-filter_complex", filter_complex_seg,
                    "-map", "0:v:0", "-map", "[a_out]", "-shortest",
                ] + std_p
                run_ffmpeg_with_progress(
                    build_ffmpeg_progress_cmd(cmd_bgm_seg, m_ts_bgm),
                    seg_total_dur, label=f"Rank {rank} Seg BGM",
                )
                # Replace m_ts with the BGM version
                os.replace(m_ts_bgm, m_ts)

        else:
            # Standard MAIN render (full start_time → end_time)
            if use_split:
                print("   📸 [Main] Split-screen render (Visual)...")
                get_x_main = buat_video_split_screen(
                    cfg.file_video_asli,
                    m_silent,
                    m_start,
                    m_end,
                    rasio,
                    diarization_data,
                    cfg,
                    label=f"Rank {rank} Main SplitScreen",
                    broll_data=broll_aktif,
                )
            elif use_camera_switch:
                # Note: Camera Switch doesn't currently support dev_mode frames but we pass it anyway
                print("   📸 [Main] Camera switch render (Visual)...")
                get_x_main = buat_video_camera_switch(
                    cfg.file_video_asli,
                    m_silent,
                    m_start,
                    m_end,
                    rasio,
                    diarization_data,
                    cfg,
                    label=f"Rank {rank} Main CameraSwitch",
                    broll_data=broll_aktif,
                )
            else:
                print("   📸 [Main] Hybrid render (Visual)...")
                get_x_main = buat_video_hybrid(
                    cfg.file_video_asli,
                    m_silent,
                    m_start,
                    m_end,
                    rasio,
                    cfg,
                    broll_aktif,
                    label=f"Rank {rank} Main",
                )

            vo_data = clip.get("voiceover")
            
            if not cfg.no_subs:
                buat_file_ass(
                    data_segmen,
                    m_start,
                    m_end,
                    a_main,
                    rasio,
                    cfg,
                    typography_plan=typography_plan,
                    gunakan_advanced=True,
                    get_x_func=get_x_main,
                    source_dim=source_dim,
                )

            print(f"   🎬 [Main] FFmpeg {'skip subtitle' if cfg.no_subs else 'burn subtitle'}...")
            esc_ass_main = escape_ffmpeg_filter_value(os.path.abspath(a_main)) if not cfg.no_subs else ""
            esc_fontsdir = escape_ffmpeg_filter_value(os.path.abspath(cfg.font_dir))

            # SMART BGM
            aktif_bgm = cfg.use_auto_bgm
            bgm_mood = clip.get("bgm_mood", "chill")
            if bgm_mood not in getattr(cfg, "bgm_moods", ["chill"]):
                bgm_mood = "chill"
                
            file_bgm = None
            if aktif_bgm:
                print(f"   🎵 Mencari file BGM lokal (Mood: {bgm_mood})...")
                file_bgm = get_local_bgm_file(bgm_mood, getattr(cfg, "bgm_dir", os.path.join(cfg.base_dir, "assets", "bgm")))
                if not file_bgm and bgm_mood != "chill":
                    print("   🔄 Fallback mencari BGM chill...")
                    file_bgm = get_local_bgm_file("chill", getattr(cfg, "bgm_dir", os.path.join(cfg.base_dir, "assets", "bgm")))
                
                if file_bgm:
                    print(f"   ✅ BGM siap: {file_bgm}")
                else:
                    print("   ⚠️ Folder BGM kosong atau file mp3 tidak ditemukan. Render lanjut tanpa BGM.")

            # --- Subtitle & BGM Encoding Loop (Handles dual output files if needed) ---
            runs = [m_silent] if not dev_dual else [m_silent, m_silent.replace(".ts", "_dev.ts")]
            out_targets = [m_ts] if not dev_dual else [m_ts, m_ts_dev]
            
            for input_silent_ts, output_final_ts in zip(runs, out_targets):
                lbl_suffix = "" if input_silent_ts == m_silent else " (DEV)"
                
                v_filter_parts = [f"subtitles={esc_ass_main}:fontsdir={esc_fontsdir}"] if not cfg.no_subs else []
                if cfg.video_sharpen:
                    v_filter_parts.append("unsharp=5:5:0.5:5:5:0.0")
                
                v_filter = ",".join(v_filter_parts) if v_filter_parts else "null"

                # Initialize FFmpeg command
                cmd_m_base = [
                    "ffmpeg", "-hide_banner", "-loglevel", "verbose", "-y",
                    "-i", input_silent_ts,
                    "-ss", str(m_start), "-to", str(m_end),
                    "-i", cfg.file_video_asli
                ]

                # Map inputs
                input_idx_bgm = -1
                
                if aktif_bgm and file_bgm:
                    cmd_m_base.extend(["-stream_loop", "-1", "-i", file_bgm])
                    input_idx_bgm = 2
                    
                if input_idx_bgm != -1:
                    # Only BGM, no VO
                    bgm_mode = getattr(cfg, "bgm_mode", "ducking")
                    audio_filter = build_bgm_filter(
                        bgm_mode, cfg.bgm_base_volume,
                        audio_input_voc="[1:a]", audio_input_bgm=f"[{input_idx_bgm}:a]"
                    )
                    filter_complex = f"[0:v]{v_filter}[v_out]; {audio_filter}"
                    
                    cmd_m_base.extend([
                        "-filter_complex", filter_complex,
                        "-map", "[v_out]", "-map", "[a_out]", "-shortest"
                    ])
                    
                else:
                    # No BGM — just original audio with subtitles
                    cmd_m_base.extend(["-map", "0:v:0", "-map", "1:a:0"])
                    if v_filter != "null":
                        cmd_m_base.extend(["-vf", v_filter])
                        
                cmd_m_base += std_p

                cmd_m = build_ffmpeg_progress_cmd(cmd_m_base, output_final_ts)
                rc_m, err_m = run_ffmpeg_with_progress(
                    cmd_m, m_end - m_start, label=f"Rank {rank} Main FFmpeg{lbl_suffix}"
                )
                if rc_m != 0:
                    raise RuntimeError(f"FFmpeg main{lbl_suffix} gagal:\n" + "\n".join(err_m))

        # VOICE-OVER INTRO GENERATION
        vo_ts = None
        vo_ts_dev = None
        vo_render_duration = 0.0
        ass_vo = None
        subtitle_files_for_qc = []
        vo_data = clip.get("voiceover")
        if vo_data and os.path.exists(vo_data["audio_path"]):
            vo_ts = os.path.join(cfg.outputs_dir, f"vo_intro_{rank}.ts")
            vo_ts_dev = os.path.join(cfg.outputs_dir, f"vo_intro_{rank}_dev.ts")
            print("   📸 [VO] Render voice-over intro with moving source footage...")

            try:
                # Dapatkan durasi asli dari mp3 menggunakan ffprobe agar tidak terpotong
                res = subprocess.run([
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", vo_data["audio_path"]
                ], stdout=subprocess.PIPE, text=True, check=True)
                vo_duration = float(res.stdout.strip()) + 0.5
            except Exception:
                vo_duration = float(vo_data["segments"][-1]["end"]) + 0.5 if vo_data.get("segments") else 5.0
            vo_render_duration = vo_duration

            source_visual_path = os.path.join(cfg.outputs_dir, f"vo_source_{rank}.mp4")
            source_visual_dur = max(float(m_end) - float(m_start), 0.1)
            visual_plan = _compute_voiceover_visual_plan(source_visual_dur, vo_duration)

            visual_choice = voiceover.resolve_commentary_visual(vo_plan, source_duration=source_visual_dur) if isinstance(vo_plan, dict) else {"mode": "continue"}
            chosen_visual_mode = str(visual_choice.get("mode") or "continue").lower().strip() or "continue"
            print(f"   🧠 [VO] Visual mode: {chosen_visual_mode}")

            try:
                if chosen_visual_mode == "continue":
                    if os.path.exists(m_silent) and not use_segments:
                        shutil.copyfile(m_silent, source_visual_path)
                    elif not _build_commentary_visual_source(
                        source_visual_path,
                        cfg.file_video_asli,
                        "continue",
                        m_start,
                        m_end,
                        vo_duration,
                        cfg=cfg,
                        ratio=rasio,
                        label=f"Rank {rank} VO Continue",
                    ):
                        raise RuntimeError("Gagal menyiapkan CONTINUE commentary visual")
                elif chosen_visual_mode == "replay":
                    replay_start = float(visual_choice.get("source_start", m_start))
                    replay_end = float(visual_choice.get("source_end", m_end))
                    if replay_end <= replay_start:
                        replay_start, replay_end = m_start, m_end
                    if not _build_commentary_visual_source(
                        source_visual_path, cfg.file_video_asli, "replay",
                        replay_start, replay_end, vo_duration,
                        cfg=cfg, ratio=rasio, label=f"Rank {rank} VO Replay",
                    ):
                        raise RuntimeError("Gagal menyiapkan replay commentary visual")
                elif chosen_visual_mode == "zoom":
                    zoom_start = float(visual_choice.get("source_start", m_start))
                    zoom_end = float(visual_choice.get("source_end", m_end))
                    if zoom_end <= zoom_start:
                        zoom_start, zoom_end = m_start, m_end
                    if not _build_commentary_visual_source(
                        source_visual_path, cfg.file_video_asli, "zoom",
                        zoom_start, zoom_end, vo_duration,
                        cfg=cfg, ratio=rasio, label=f"Rank {rank} VO Zoom",
                    ):
                        raise RuntimeError("Gagal menyiapkan zoom commentary visual")
                elif chosen_visual_mode == "broll" and getattr(cfg, "use_broll", False):
                    broll_query = broll_policy.normalize_broll_query(visual_choice.get("query"))
                    broll_intent = broll_policy.normalize_visual_intent(visual_choice.get("visual_intent"))
                    broll_temp = os.path.join(cfg.outputs_dir, f"vo_broll_{rank}.mp4")
                    broll_ready = bool(
                        broll_query
                        and broll_intent
                        and download_pexels_broll(
                            broll_query,
                            rasio,
                            broll_temp,
                            getattr(cfg, "pexels_api_key", None),
                            minimum_duration=vo_duration + 0.25,
                            output_dir=cfg.outputs_dir,
                            visual_intent=broll_intent,
                        )
                    )
                    if broll_ready:
                        source_visual_path = broll_temp
                        asset_metadata = broll_policy.get_asset_metadata(cfg.outputs_dir, broll_temp)
                        if asset_metadata:
                            manifest_item["broll_assets"].append(asset_metadata)
                    else:
                        print("   ⚠️ [VO] B-roll unavailable or invalid; falling back to CONTINUE.")
                        manifest_item["broll_fallbacks"].append({
                            "query": str(visual_choice.get("query") or ""),
                            "reason": "Pexels search/download/media validation failed",
                        })
                        if not _build_commentary_visual_source(
                            source_visual_path, cfg.file_video_asli, "continue",
                            m_start, m_end, vo_duration,
                            cfg=cfg, ratio=rasio, label=f"Rank {rank} VO B-roll fallback",
                        ):
                            raise RuntimeError("Gagal menyiapkan B-roll fallback visual")
                else:
                    if not _build_commentary_visual_source(
                        source_visual_path, cfg.file_video_asli, "continue",
                        m_start, m_end, vo_duration,
                        cfg=cfg, ratio=rasio, label=f"Rank {rank} VO visual fallback",
                    ):
                        raise RuntimeError("Gagal menyiapkan source visual fallback")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Gagal menyiapkan visual bergerak untuk VO Intro:\n{e.stderr}")

            # Generate for both normal and dev dual if needed
            out_targets_vo = [vo_ts] if not dev_dual else [vo_ts, vo_ts_dev]

            for output_vo_ts in out_targets_vo:
                vo_w, vo_h = _get_render_dims(cfg, rasio, source_h=sh)
                if dev_dual and output_vo_ts == vo_ts_dev:
                    vo_w, vo_h = 1920, 1080
                elif getattr(cfg, "dev_mode_with_output_merge", False):
                    vo_w, vo_h = 2648, 1220
                elif getattr(cfg, "dev_mode", False) and not dev_dual:
                    vo_w, vo_h = 1920, 1080

                ass_vo_filter = ""
                overlay_out = "[v_out]"

                # Buat file ASS subtitle khusus untuk VO intro jika ada segments
                if not cfg.no_subs and vo_data.get("segments"):
                    ass_vo = os.path.join(cfg.outputs_dir, f"vo_subs_{rank}.ass")
                    buat_file_ass(
                        vo_data["segments"],
                        0.0,
                        vo_duration,
                        ass_vo,
                        rasio,
                        cfg,
                        typography_plan=typography_plan,
                        gunakan_advanced=True,
                        get_x_func=get_x_main,
                        source_dim=(vo_w, vo_h)
                    )
                    esc_ass_vo = escape_ffmpeg_filter_value(os.path.abspath(ass_vo))
                    esc_fontsdir_vo = escape_ffmpeg_filter_value(os.path.abspath(cfg.font_dir))
                    ass_vo_filter = f"; [v_over]subtitles={esc_ass_vo}:fontsdir={esc_fontsdir_vo}[v_out]"
                    overlay_out = "[v_over]"

                # Generate edge glow overlay video
                glow_path = os.path.join(cfg.outputs_dir, f"vo_glow_{rank}.mp4")
                glow_mode = getattr(cfg, "edge_glow_mode", "smooth")
                print(f"   ✨ [VO] Generating ambient edge glow (mode={glow_mode})...")

                if glow_mode == "full":
                    glow_dur = vo_duration
                    glow_seamless = False
                    glow_needs_loop = False
                elif glow_mode == "smooth":
                    glow_dur = min(10.0, vo_duration)
                    glow_seamless = True
                    glow_needs_loop = glow_dur < vo_duration
                else:
                    glow_dur = min(10.0, vo_duration)
                    glow_seamless = False
                    glow_needs_loop = glow_dur < vo_duration

                generate_edge_glow_video(
                    glow_path, vo_w, vo_h,
                    duration=glow_dur,
                    fps=30,
                    glow_speed=0.15,
                    opacity=0.45,
                    seamless_loop=glow_seamless,
                )

                # =========================
                # Manual Wave/Spectrum Config
                # =========================
                wave_enabled = True
                wave_w = 800
                wave_h = 260
                wave_rate = 30
                wave_lowpass = 300
                wave_use_lowpass = True
                wave_mode = "cline"
                wave_color = "0x00FFFF"
                wave_scale = "sqrt"
                wave_alpha = 0.65
                wave_x = "(W-w)/2"
                wave_y = "(H-h)/2"
                wave_key_color = "0x000000"
                wave_key_similarity = 0.1
                wave_key_blend = 0.1

                if wave_use_lowpass:
                    wave_audio_filter = f"[vo_wave_in]lowpass=f={wave_lowpass}"
                else:
                    wave_audio_filter = "[vo_wave_in]anull"

                wave_filter = (
                    f"{wave_audio_filter},"
                    f"showwaves="
                    f"s={wave_w}x{wave_h}:"
                    f"mode={wave_mode}:"
                    f"colors={wave_color}:"
                    f"rate={wave_rate}:"
                    f"scale={wave_scale},"
                    f"format=rgba,"
                    f"colorkey={wave_key_color}:{wave_key_similarity}:{wave_key_blend},"
                    f"colorchannelmixer=aa={wave_alpha}"
                    f"[wave_v]; "
                )

                v_filter_vo = (
                    f"[0:v]scale={vo_w}:{vo_h}:force_original_aspect_ratio=increase,"
                    f"crop={vo_w}:{vo_h},"
                    f"colorchannelmixer=rr=0.3:gg=0.3:bb=0.3[v_bg]; "
                    f"[2:v]scale={vo_w}:{vo_h}[glow_scaled]; "
                    f"[v_bg][glow_scaled]blend=all_mode=screen:shortest=1[v_glowed]; "
                    f"[1:a]asplit=2[vo_a][vo_wave_in]; "
                    f"{wave_filter}"
                    f"[v_glowed][wave_v]overlay={wave_x}:{wave_y}:shortest=1{overlay_out}"
                    f"{ass_vo_filter}"
                )

                cmd_vo_base = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                ]
                if visual_plan["loop_required"]:
                    cmd_vo_base.extend(["-stream_loop", "-1", "-i", source_visual_path])
                else:
                    cmd_vo_base.extend(["-i", source_visual_path])
                cmd_vo_base.extend(["-i", vo_data["audio_path"]])

                if glow_needs_loop:
                    cmd_vo_base.extend(["-stream_loop", "-1", "-i", glow_path])
                else:
                    cmd_vo_base.extend(["-i", glow_path])

                if aktif_bgm and file_bgm:
                    cmd_vo_base.extend(["-stream_loop", "-1", "-i", file_bgm])
                    bgm_vol = cfg.bgm_base_volume
                    vo_vol = getattr(cfg, "voiceover_volume", 1.0)
                    vo_fade_filter = voiceover.build_commentary_audio_filter(
                        source_audio_label=None,
                        narration_audio_label="[vo_a]",
                        original_volume=getattr(cfg, "original_volume", 0.15),
                        voiceover_volume=vo_vol,
                        narration_duration=vo_duration,
                    )
                    audio_filter_vo = (
                        f"[3:a]volume={bgm_vol}[bgm_vol]; "
                        f"{vo_fade_filter}; "
                        f"[bgm_vol][a_out]amix=inputs=2:duration=first:dropout_transition=2[a_out_final]"
                    )
                    v_filter_vo += f"; {audio_filter_vo}"
                else:
                    vo_vol = getattr(cfg, "voiceover_volume", 1.0)
                    vo_fade_filter = voiceover.build_commentary_audio_filter(
                        source_audio_label=None,
                        narration_audio_label="[vo_a]",
                        original_volume=getattr(cfg, "original_volume", 0.15),
                        voiceover_volume=vo_vol,
                        narration_duration=vo_duration,
                    )
                    v_filter_vo += f"; {vo_fade_filter}"

                cmd_vo_base.extend([
                    "-filter_complex", v_filter_vo,
                    "-map", "[v_out]", "-map", "[a_out]", "-t", str(vo_duration)
                ])
                cmd_vo_base += std_p
                cmd_vo_base.append(output_vo_ts)
                
                try:
                    subprocess.run(cmd_vo_base, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"FFmpeg VO intro gagal (Rank {rank}):\nCommand: {' '.join(cmd_vo_base)}\nError:\n{e.stderr}")
                
            if os.path.exists(source_visual_path):
                os.remove(source_visual_path)
            if os.path.exists(glow_path):
                os.remove(glow_path)


        # FINAL CONCAT
        print("   🔗 [Final] Menyelesaikan clip akhir...")
        
        # Calculate target dimensions for each run
        out_w_std, out_h_std = _get_render_dims(cfg, rasio, source_h=sh)
        if getattr(cfg, "dev_mode_with_output_merge", False):
            out_w_std, out_h_std = 2648, 1220
        elif getattr(cfg, "dev_mode", False) and not dev_dual:
            # Single stream pure dev mode
            out_w_std, out_h_std = 1920, 1080
            
        concat_runs = [(out_vid, m_ts, h_ts, (out_w_std, out_h_std))]
        if dev_dual:
            out_vid_dev = os.path.join(cfg.outputs_dir, f"highlight_rank_{rank}_dev_mode_ready.mp4")
            # Usually hook doesn't generate dual, so we fallback to standard hook for dev if missing
            h_dev_target = h_ts_dev if os.path.exists(h_ts_dev) else h_ts 
            concat_runs.append((out_vid_dev, m_ts_dev, h_dev_target, (1920, 1080)))
            
        for final_path, main_vid_ts, hook_vid_ts, dims in concat_runs:
            vo_vid_ts = vo_ts_dev if (dev_dual and final_path.endswith("_dev_mode_ready.mp4")) else vo_ts
            
            # Determine concat strategy
            parts = []
            
            # 1. Hook
            if use_hook_v2 and os.path.exists(hook_vid_ts):
                parts.append(hook_vid_ts)
            elif aktif_hook and os.path.exists(hook_vid_ts):
                parts.append(hook_vid_ts)
                cur_glitch = siapkan_glitch_video(rasio, cfg, video_encoder, source_h=sh, custom_dims=dims)
                if cur_glitch and os.path.exists(cur_glitch):
                    parts.append(cur_glitch)
                    
            # 2. Voice-Over Intro
            if vo_vid_ts and os.path.exists(vo_vid_ts):
                parts.append(vo_vid_ts)
                
            # 3. Main Clip
            parts.append(main_vid_ts)

            concat_str = "concat:" + "|".join(parts)

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    concat_str,
                    "-c",
                    "copy",
                    "-bsf:a",
                    "aac_adtstoasc",
                    final_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        # EDGE GLOW POST-PROCESSING (full clip)
        if getattr(cfg, "edge_glow", False):
            print("   ✨ [Edge Glow] Applying ambient edge glow to full clip...")
            for final_path, _, _, dims in concat_runs:
                if not os.path.exists(final_path):
                    continue
                gw, gh = dims
                # Get video duration via ffprobe
                try:
                    dur_res = subprocess.run([
                        "ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", final_path
                    ], stdout=subprocess.PIPE, text=True, check=True)
                    clip_dur = float(dur_res.stdout.strip())
                except Exception:
                    clip_dur = 60.0

                glow_full_path = os.path.join(cfg.outputs_dir, f"glow_full_{rank}.mp4")
                glow_mode_fc = getattr(cfg, "edge_glow_mode", "smooth")

                if glow_mode_fc == "full":
                    glow_fc_dur = clip_dur
                    glow_fc_seamless = False
                    glow_fc_needs_loop = False
                elif glow_mode_fc == "smooth":
                    glow_fc_dur = min(10.0, clip_dur)
                    glow_fc_seamless = True
                    glow_fc_needs_loop = glow_fc_dur < clip_dur
                else:  # "default"
                    glow_fc_dur = min(10.0, clip_dur)
                    glow_fc_seamless = False
                    glow_fc_needs_loop = glow_fc_dur < clip_dur

                generate_edge_glow_video(
                    glow_full_path, gw, gh,
                    duration=glow_fc_dur,
                    fps=30,
                    glow_speed=0.15,
                    opacity=0.45,
                    seamless_loop=glow_fc_seamless,
                )

                # Overlay via FFmpeg blend=screen
                tmp_glowed = final_path + ".glowed.mp4"
                cmd_glow = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", final_path,
                ] + (["-stream_loop", "-1"] if glow_fc_needs_loop else []) + [
                    "-i", glow_full_path,
                    "-filter_complex",
                    f"[1:v]scale={gw}:{gh}[glow]; [0:v][glow]blend=all_mode=screen:shortest=1[v_out]",
                    "-map", "[v_out]", "-map", "0:a?",
                    "-c:a", "copy",
                    "-t", str(clip_dur),
                ]
                cmd_glow += std_p
                cmd_glow.append(tmp_glowed)

                try:
                    subprocess.run(cmd_glow, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                    os.replace(tmp_glowed, final_path)
                    print(f"   ✅ [Edge Glow] Applied to {os.path.basename(final_path)}")
                except subprocess.CalledProcessError as e:
                    print(f"   ⚠️ [Edge Glow] Failed for {os.path.basename(final_path)}: {e.stderr[-300:]}")
                    if os.path.exists(tmp_glowed):
                        os.remove(tmp_glowed)
                finally:
                    if os.path.exists(glow_full_path):
                        os.remove(glow_full_path)

        thumbnail_text = final_metadata.choose_thumbnail_text(clip)
        judul_thumbnail = thumbnail_text or judul_en or judul or f"Highlight {rank}"
        audio_mastering_result = audio_mastering.master_audio_in_place(out_vid)
        manifest_item["audio_mastering"] = audio_mastering_result
        thumbnail_source_time = buat_thumbnail(
            out_vid,
            out_thm,
            judul_thumbnail,
            cfg,
        )
        manifest_item["thumbnail_text"] = judul_thumbnail
        manifest_item["thumbnail_source_time"] = thumbnail_source_time

        vo_enabled = bool(getattr(cfg, "voiceover", False) and isinstance(vo_data, dict) and vo_data.get("audio_path"))
        for subtitle_path in (a_hook, a_main, ass_vo):
            if subtitle_path and os.path.exists(subtitle_path):
                subtitle_files_for_qc.append(subtitle_path)
        expected_output_duration = (
            pacing_result["output_duration"]
            + hook_output_offset
            + vo_render_duration
        )
        final_pacing_map = pacing.shift_output_map(
            pacing_result["map"],
            hook_output_offset + vo_render_duration,
        )
        pacing_validation = pacing.validate_pacing_map(
            final_pacing_map,
            expected_output_duration,
        )
        manifest_item["pacing_final_map"] = final_pacing_map

        qc_result = voiceover.validate_final_output(
            out_vid,
            expected_duration=expected_output_duration,
            expected_ratio=getattr(cfg, "pilihan_rasio", None),
            voiceover_enabled=vo_enabled,
            commentary_plan=vo_plan if isinstance(vo_plan, dict) else None,
            commentary_audio_paths=[vo_data["audio_path"]] if vo_enabled and isinstance(vo_data, dict) and vo_data.get("audio_path") else None,
            subtitle_paths=subtitle_files_for_qc or None,
            caption_segments=vo_data.get("segments") if vo_enabled and isinstance(vo_data, dict) else None,
        )
        if manifest_item["broll_fallbacks"]:
            qc_result["warnings"].append(
                f"{len(manifest_item['broll_fallbacks'])} B-roll request(s) fell back to source footage."
            )
            if qc_result["status"] == "pass":
                qc_result["status"] = "warning"
        if audio_mastering_result.get("status") == "fallback":
            qc_result["warnings"].append(
                "Final audio mastering fell back to the pre-master mix."
            )
            if qc_result["status"] == "pass":
                qc_result["status"] = "warning"
        if audio_mastering_result.get("status") == "mastered":
            post_audio = audio_mastering_result.get("post") or {}
            measured_lufs = post_audio.get("integrated_lufs")
            measured_peak = post_audio.get("true_peak_db")
            loudness_ok = measured_lufs is not None and abs(measured_lufs - audio_mastering.TARGET_I) <= 1.5
            peak_ok = measured_peak is not None and measured_peak <= audio_mastering.TARGET_TP + 0.2
            qc_result["checks"]["audio_loudness"] = "pass" if loudness_ok else "warning"
            qc_result["checks"]["audio_true_peak"] = "pass" if peak_ok else "warning"
            if not loudness_ok or not peak_ok:
                qc_result["warnings"].append("Final audio measurements are outside the preferred mastering tolerance.")
                if qc_result["status"] == "pass":
                    qc_result["status"] = "warning"
        qc_result["checks"]["pacing_timeline"] = "pass" if pacing_validation["valid"] else "fail"
        if not pacing_validation["valid"]:
            qc_result["status"] = "fail"
            qc_result["errors"].extend(pacing_validation["errors"])
        manifest_item["quality_control"] = qc_result
        manifest_item["quality_status"] = qc_result["status"]
        if qc_result["status"] == "fail":
            manifest_item["status"] = "failed"
            print(f"⚠️ [Rank {rank}] Final QC failed: {'; '.join(qc_result['errors'])[:250]}")
        elif qc_result["status"] == "warning":
            manifest_item["status"] = "success"
            print(f"⚠️ [Rank {rank}] Final QC warning: {'; '.join(qc_result['warnings'])[:250]}")
        else:
            manifest_item["status"] = "success"
            print(f"✅ [Rank {rank}] Final QC passed.")

        manifest_item["video_exists"] = os.path.exists(out_vid)
        manifest_item["thumbnail_exists"] = os.path.exists(out_thm)

        print(f"✅ [Rank {rank}] Selesai.")
        return manifest_item

    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: FFmpeg gagal. Error: {e}")
        manifest_item["status"] = "failed"
        manifest_item["error"] = str(e)
        manifest_item["video_exists"] = os.path.exists(out_vid)
        manifest_item["thumbnail_exists"] = os.path.exists(out_thm)
        return manifest_item

    except Exception as e:
        print(f"\n❌ ERROR: Kegagalan tak terduga. Error: {e}")
        manifest_item["status"] = "failed"
        manifest_item["error"] = str(e)
        manifest_item["video_exists"] = os.path.exists(out_vid)
        manifest_item["thumbnail_exists"] = os.path.exists(out_thm)
        return manifest_item

    finally:
        files_to_remove = [h_ts, m_ts, a_hook, a_main, h_silent, m_silent]
        if dev_dual:
            files_to_remove.extend([h_ts_dev, m_ts_dev, m_silent.replace(".ts", "_dev.ts")])
            
        for br in broll_aktif:
            files_to_remove.append(br["filepath"])

        for f_path in files_to_remove:
            if os.path.exists(f_path):
                os.remove(f_path)
