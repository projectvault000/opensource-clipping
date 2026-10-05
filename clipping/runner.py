"""
clipping.runner — Pipeline Orchestrator

Maps to Cell 4 (Execute) of the notebook.
Orchestrates the full clip generation pipeline.
"""

import hashlib
import json
import math
import os
import subprocess
import time

from . import diarization as diarization_mod
from . import broll_policy, engine, export_package, final_metadata, metadata, studio, hook_manager, voiceover
from .cache_manager import CacheManager
from .checkpoints import JobCheckpoint, prepare_cli_job, render_manifest_complete
from .review_manager import ReviewManager
from .retry import normalize_retry_plan, save_run_config


def _probe_video_duration(video_path: str) -> float | None:
    """Return media duration when ffprobe is available, otherwise defer validation."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=15,
        )
        duration = float(result.stdout.strip())
        return duration if math.isfinite(duration) and duration > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def run_pipeline(cfg, whisper_model_instance=None, *, target_rank: int | None = None) -> list[dict]:
    """
    Run the full clipping pipeline:
      1. Download YouTube video
      2. Transcribe with Whisper
      3. Analyse with Gemini AI
      4. Normalize metadata
      5. Prepare glitch transition
      6. Render each clip
      7. Save render_manifest.json

    The run is checkpoint-aware: stage completion is persisted and re-used when
    the same job directory is resumed after an interruption.

    Parameters
    ----------
    cfg : SimpleNamespace
        Configuration object from ``config.build_config()``.
    whisper_model_instance : WhisperModel or None
        Optional pre-loaded WhisperModel to reuse across batch runs.

    Returns
    -------
    list[dict]
        Render manifest (one dict per clip).
    """

    prepare_cli_job(cfg)
    checkpoint = JobCheckpoint.from_cfg(cfg)
    cfg.job_id = checkpoint.job_id
    if target_rank is None:
        save_run_config(cfg)
    cache_manager = CacheManager(getattr(cfg, "cache_dir", getattr(cfg, "outputs_dir", "outputs")))
    broll_policy.reset_broll_session(cfg.outputs_dir)

    manifest_path = os.path.join(cfg.outputs_dir, "render_manifest.json")
    if target_rank is not None:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
        if not any(int(row.get("rank", 0)) == target_rank for row in existing_manifest):
            raise ValueError(f"Clip rank {target_rank} is absent from the job manifest.")
    elif os.path.exists(manifest_path) and checkpoint.is_complete("render"):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            existing_manifest = json.load(handle)
        if render_manifest_complete(existing_manifest, cfg.jumlah_clip):
            print(f"Resume checkpoint detected for {checkpoint.job_id}; reusing {manifest_path}")
            return existing_manifest

    # Step 1 — Download
    source_platform = getattr(cfg, "source_platform", "youtube")
    if os.path.exists(cfg.file_video_asli) and checkpoint.is_complete("download"):
        print(f"✅ Reusing existing source video for job {checkpoint.job_id}: {cfg.file_video_asli}")
    else:
        checkpoint.mark_started("download", artifact_path=cfg.file_video_asli)
        try:
            engine.download_video(
                cfg.url_youtube,
                cfg.file_video_asli,
                getattr(cfg, "use_dlp_subs", False),
                getattr(cfg, "download_source_height", "max"),
                source_platform=source_platform,
                yt_cookies=getattr(cfg, "yt_cookies", None),
            )
            checkpoint.mark_succeeded("download", artifact_path=cfg.file_video_asli)
        except Exception:
            checkpoint.mark_failed("download", "download failed")
            raise

    # Step 2 — Transcribe
    transkrip_lengkap = ""
    data_segmen = []
    transcript_cache_key = CacheManager.make_key(
        "transcript",
        source_url=getattr(cfg, "url_youtube", None) or cfg.file_video_asli,
        source_file=cfg.file_video_asli,
        whisper_model=cfg.whisper_model,
        whisper_device=cfg.whisper_device,
        whisper_compute_type=cfg.whisper_compute_type,
        max_words_per_subtitle=getattr(cfg, "max_kata_per_subtitle", 5),
        source_platform=source_platform,
    )

    transcribe_started = time.perf_counter()
    cached_transcript = cache_manager.get_json("transcript", transcript_cache_key)
    if cached_transcript and cached_transcript.get("text") and cached_transcript.get("segments"):
        transkrip_lengkap = cached_transcript["text"]
        data_segmen = cached_transcript["segments"]
        print(f"✅ Reused transcript cache for {cfg.file_video_asli} ({transcript_cache_key})")
    else:
        checkpoint.mark_started("transcribe")
        try:
            import glob

            json3_files = glob.glob(cfg.file_video_asli.replace(".mp4", ".*.json3"))
            file_json3 = json3_files[0] if json3_files else None
            if source_platform == "youtube" and getattr(cfg, "use_dlp_subs", False) and file_json3:
                transkrip_lengkap, data_segmen = engine.parse_youtube_json3_subs(
                    file_json3, max_words_per_subtitle=cfg.max_kata_per_subtitle
                )
            if not transkrip_lengkap or not data_segmen:
                transkrip_lengkap, data_segmen = engine.transcribe_video(
                    cfg.file_video_asli,
                    max_words_per_subtitle=cfg.max_kata_per_subtitle,
                    model_size=cfg.whisper_model,
                    device=cfg.whisper_device,
                    compute_type=cfg.whisper_compute_type,
                    whisper_model_instance=whisper_model_instance,
                )
            cache_manager.put_json(
                "transcript", transcript_cache_key,
                {"text": transkrip_lengkap, "segments": data_segmen},
            )
            checkpoint.mark_succeeded("transcribe", cache_key=transcript_cache_key)
        except Exception as exc:
            checkpoint.mark_failed("transcribe", str(exc))
            raise
    cache_manager.record_timing("transcribe", time.perf_counter() - transcribe_started)

    # Step 3 — Gemini AI analysis
    gemini_output_path = os.path.join(cfg.outputs_dir, "gemini_response.json")
    
    gemini_cache_key = CacheManager.make_key(
        "gemini",
        source_url=getattr(cfg, "url_youtube", None) or cfg.file_video_asli,
        audio_file=cfg.file_video_asli,
        transcript_hash=hashlib.sha256((transkrip_lengkap or "").encode("utf-8")).hexdigest(),
        ai_provider=getattr(cfg, "ai_provider", "gemini"),
        model=getattr(cfg, "ai_model", None),
        custom_prompt=getattr(cfg, "prompt_template", None),
    )

    if target_rank is not None:
        with open(os.path.join(cfg.outputs_dir, "metadata_preview.json"), "r", encoding="utf-8") as handle:
            saved_plan = json.load(handle)
        hasil_json = saved_plan
    elif getattr(cfg, "load_gemini_json", False) and os.path.exists(gemini_output_path):
        print(f"\n🔄 [3/3] Memuat data AI ({cfg.ai_provider}) dari file lokal: {gemini_output_path}")
        with open(gemini_output_path, "r", encoding="utf-8") as f:
            hasil_json = json.load(f)
    else:
        cached_ai = cache_manager.get_json("gemini", gemini_cache_key)
        if cached_ai is not None:
            hasil_json = cached_ai
            print(f"✅ Reused Gemini analysis cache ({gemini_cache_key})")
        else:
            start_time = time.perf_counter()
            hasil_json = engine.analyze_with_ai(transkrip_lengkap, cfg)
            cache_manager.record_timing("gemini", time.perf_counter() - start_time)
            cache_manager.put_json("gemini", gemini_cache_key, hasil_json)

            # Save raw gemini json for future loading/reproduction
            with open(gemini_output_path, "w", encoding="utf-8") as f:
                json.dump(hasil_json, f, indent=4, ensure_ascii=False)
            print(f"💾 Raw AI response tersimpan di: {gemini_output_path}")

    # Step 4 — Metadata normalisation
    source_duration = _probe_video_duration(cfg.file_video_asli)
    if source_duration is None:
        print("⚠️ Source duration unavailable; clip ranges will be checked against timestamps and duration limits only.")
    if target_rank is not None:
        hasil_json = normalize_retry_plan(
            hasil_json, target_rank, source_duration,
            engine.MIN_CLIP_DURATION, engine.MAX_CLIP_DURATION,
        )
    else:
        hasil_json = metadata.normalize_and_validate(
            hasil_json,
            source_duration=source_duration,
            min_duration=engine.MIN_CLIP_DURATION,
            max_duration=engine.MAX_CLIP_DURATION,
        )
    metadata.print_preview(hasil_json)

    metadata_path = os.path.join(cfg.outputs_dir, "metadata_preview.json")
    if target_rank is None:
        metadata.save_metadata_preview(hasil_json, path=metadata_path)

    # Step 5 — Diarization (split-screen / camera-switch)
    diarization_data = None
    if (
        (getattr(cfg, "use_split_screen", False) and cfg.split_trigger == "diarization")
        or getattr(cfg, "use_camera_switch", False)
    ) and studio._is_vertical_ratio(cfg.pilihan_rasio):
        try:
            mode_label = (
                "Split-Screen"
                if getattr(cfg, "use_split_screen", False)
                else "Camera-Switch"
            )
            print(f"\n🎙️ [{mode_label}] Menjalankan speaker diarization...")
            audio_path = cfg.file_video_asli.replace(".mp4", "_audio.wav")
            diarization_mod.extract_audio(cfg.file_video_asli, audio_path)
            num_speakers_arg = getattr(cfg, "diarization_num_speakers", 2)
            min_spk = None
            max_spk = None

            if str(num_speakers_arg).lower() == "auto":
                max_faces = studio.estimate_speaker_count_from_video(
                    cfg.file_video_asli, cfg
                )
                num_speakers_arg = "auto"
                min_spk = max(1, max_faces)
                max_spk = min_spk + 2
                print(f"   ℹ️ Instruksi Pyannote: {min_spk} hingga {max_spk} speaker.")

            diarization_data = diarization_mod.run_diarization(
                audio_path,
                hf_token=cfg.hf_token,
                num_speakers=num_speakers_arg,
                min_speakers=min_spk,
                max_speakers=max_spk,
            )
            # Clean up temp audio
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"⚠️ Diarization gagal: {e}")
            print("   Fallback ke mode render biasa (tanpa split-screen).")
            diarization_data = None

    # Step 6 — Video encoder & glitch
    os.environ["OSC_VIDEO_SCALE_ALGO"] = str(
        getattr(cfg, "video_scale_algo", "lanczos")
    )
    
    # Get target dimensions for auto-bitrate calculation
    import cv2
    cap_e = cv2.VideoCapture(cfg.file_video_asli)
    src_h_e = int(cap_e.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_e.release()
    
    target_w_e, target_h_e = studio._get_render_dims(cfg, cfg.pilihan_rasio, source_h=src_h_e)
    video_encoder = studio.detect_video_encoder(cfg, target_h=target_h_e)

    file_glitch_ts = None
    if cfg.use_hook_glitch:
        print("⚙️ Menyiapkan Video Glitch Transisi...")
        
        # Get source dimensions for proper glitch scaling
        import cv2
        cap_g = cv2.VideoCapture(cfg.file_video_asli)
        source_h_g = int(cap_g.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_g.release()

        file_glitch_ts = studio.siapkan_glitch_video(
            cfg.pilihan_rasio, cfg, video_encoder, source_h=source_h_g
        )

    # Step 6 — Render each clip
    render_manifest: list[dict] = []

    custom_hook_path = None
    if getattr(cfg, "hook_source", None):
        print("\n🎣 Mengunduh sumber klip Hook kustom...")
        custom_hook_path = hook_manager.download_custom_hook(cfg)

    # Step 5.5 — Generate Voice-Over (if enabled)
    if getattr(cfg, "voiceover", False):
        print(f"\n🎙️ Meng-generate Voice-Over untuk {len(hasil_json)} klip...")
        for klip in hasil_json:
            try:
                # 1. Generate commentary script from snippet
                start = float(klip["start_time"])
                end = float(klip["end_time"])
                # Extract transcript snippet for this time range
                snippet_lines = []
                for seg in data_segmen:
                    if float(seg["end"]) > start and float(seg["start"]) < end:
                        # Support both Whisper format (has 'text') and YouTube JSON3 (only 'words')
                        seg_text = seg.get("text") or " ".join(w["word"] for w in seg.get("words", []))
                        if seg_text:
                            snippet_lines.append(seg_text)
                snippet_text = " ".join(snippet_lines)

                clip_duration = max(float(end) - float(start), 1.0)
                plan = voiceover.generate_commentary_plan(
                    snippet_text,
                    cfg,
                    style=cfg.voiceover_style,
                    language=cfg.voiceover_lang,
                    length=cfg.voiceover_length,
                    clip_duration=clip_duration,
                )
                timeline = voiceover.build_commentary_timeline(
                    plan,
                    clip_duration=clip_duration,
                    initial_output_offset=voiceover.estimate_hook_output_offset(
                        klip,
                        cfg,
                        legacy_transition_duration=(
                            1.0
                            if not getattr(cfg, "hook_v2", False)
                            and file_glitch_ts
                            and os.path.exists(file_glitch_ts)
                            else 0.0
                        ),
                    ),
                )
                script = voiceover._extract_narration_from_plan(plan)
                if not script:
                    script = voiceover.generate_commentary_script(
                        snippet_text,
                        cfg,
                        style=cfg.voiceover_style,
                        language=cfg.voiceover_lang,
                        length=cfg.voiceover_length,
                    )

                hook_plan = klip.get("hook_plan", {})
                if (
                    isinstance(hook_plan, dict)
                    and hook_plan.get("type") == "commentary_hook"
                    and hook_plan.get("text")
                ):
                    script = f"{hook_plan['text'].strip()} {script}".strip()

                if script:
                    voiceover_key = CacheManager.make_key(
                        "voiceover",
                        text=script,
                        voice=getattr(cfg, "voiceover_voice", None),
                        voice_lang=getattr(cfg, "voiceover_lang", None),
                        voice_style=getattr(cfg, "voiceover_style", None),
                        voice_length=getattr(cfg, "voiceover_length", None),
                        ratio=getattr(cfg, "pilihan_rasio", None),
                        rank=klip.get("rank"),
                    )
                    cached_voice = cache_manager.get_json("voiceover", voiceover_key)
                    if cached_voice and cached_voice.get("audio_path") and os.path.exists(cached_voice["audio_path"]):
                        audio_path = cached_voice["audio_path"]
                        vo_segments = cached_voice.get("segments", [])
                        print(f"✅ Reused voice-over cache for rank {klip['rank']} ({voiceover_key})")
                    else:
                        start_time = time.perf_counter()
                        audio_path, vo_segments = voiceover.synthesize_voice(
                            script,
                            cfg.voiceover_voice,
                            cache_manager.category_dir("voiceover"),
                            voiceover_key,
                            max_words_per_subtitle=getattr(cfg, "max_kata_per_subtitle", 5),
                        )
                        cache_manager.record_timing("voiceover", time.perf_counter() - start_time)
                        cache_manager.put_json("voiceover", voiceover_key, {"audio_path": audio_path, "segments": vo_segments, "voice": cfg.voiceover_voice})

                    if os.path.exists(audio_path):
                        klip["voiceover"] = {
                            "script": script,
                            "audio_path": audio_path,
                            "segments": vo_segments,
                            "voice": cfg.voiceover_voice,
                            "plan": plan,
                            "timeline": timeline,
                        }

            except Exception as e:
                print(f"   ⚠️ Gagal generate voice-over untuk Rank {klip['rank']}: {e}")

    for klip in sorted(hasil_json, key=lambda x: x["rank"]):
        
        if custom_hook_path:
            klip["custom_hook_info"] = {"file_path": custom_hook_path}

        hasil_render = studio.proses_klip(
            klip["rank"],
            klip,
            cfg.pilihan_rasio,
            file_glitch_ts,
            data_segmen,
            cfg,
            video_encoder,
            diarization_data=diarization_data,
        )
        if hasil_render:
            render_manifest.append(hasil_render)

    if target_rank is not None:
        if len(render_manifest) != 1 or render_manifest[0].get("status") != "success":
            raise RuntimeError(f"Clip {target_rank} failed to rerender; existing manifest was not replaced.")
        refreshed = render_manifest[0]
        refreshed["generation_version"] = 1 + int(next(
            row.get("generation_version", 1) for row in existing_manifest
            if int(row.get("rank", 0)) == target_rank
        ))
        render_manifest = [
            refreshed if int(row.get("rank", 0)) == target_rank else row
            for row in existing_manifest
        ]

    # Step 7 — Inject source metadata for attribution & safety tracking
    for row in render_manifest:
        if target_rank is not None and int(row.get("rank", 0)) != target_rank:
            continue
        # Attach source URL so metadata.py can auto-add source credit
        if not row.get("source_url"):
            row["source_url"] = getattr(cfg, "url_youtube", None)
        metadata_path = os.path.join(
            cfg.outputs_dir,
            f"clip_{row.get('rank', 'unknown')}_metadata.json",
        )
        try:
            final_metadata.write_metadata_sidecar(row, metadata_path)
            row["metadata_path"] = metadata_path
        except (OSError, TypeError, ValueError) as exc:
            row["metadata_error"] = str(exc)

    package_summary = export_package.export_publishing_package(
        render_manifest,
        cfg.outputs_dir,
        source_segments=data_segmen,
        target_rank=target_rank,
    )
    print(
        f"[Export] Packages created: {package_summary['pass']} PASS, "
        f"{package_summary['warning']} WARNING, {package_summary['fail']} FAIL."
    )
    for row in render_manifest:
        package_id = f"clip_{int(row.get('rank', 0) or 0):02d}"
        row["publishing_package_path"] = os.path.join(
            cfg.outputs_dir, "publishing_package", package_id
        )
    # Step 8 — Save manifest
    manifest_path = os.path.join(cfg.outputs_dir, "render_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(render_manifest, f, ensure_ascii=False, indent=2)
    review_manager = ReviewManager(cfg.outputs_dir)
    render_manifest = review_manager.sync_manifest(render_manifest)
    checkpoint.mark_started("render", manifest_path=manifest_path)
    checkpoint.mark_succeeded("render", manifest_path=manifest_path, clip_count=len(render_manifest))

    cache_stats = cache_manager.get_stats()
    checkpoint.mark_started("performance", cache_stats=cache_stats)
    checkpoint.mark_succeeded("performance", cache_stats=cache_stats)

    print(f"\n💾 Render manifest disimpan ke {manifest_path} ({len(render_manifest)} item)")
    if cache_stats:
        summary = ", ".join(
            f"{name}:{values.get('hits', 0)} hit/{values.get('misses', 0)} miss"
            for name, values in cache_stats.items()
            if isinstance(values, dict) and "hits" in values and "misses" in values
        )
        timings = ", ".join(f"{name}={value:.2f}s" for name, value in cache_stats.get("stage_timings", {}).items())
        print(f"⚡ Cache summary: {summary} | timings: {timings or 'n/a'}")

    pass_count = sum(1 for item in render_manifest if str(item.get("quality_status") or item.get("status") or "success").lower() == "pass")
    warn_count = sum(1 for item in render_manifest if str(item.get("quality_status") or item.get("status") or "success").lower() == "warning")
    fail_count = sum(1 for item in render_manifest if str(item.get("quality_status") or item.get("status") or "success").lower() in {"fail", "failed"})

    print("\n================================")
    print("FINAL QUALITY CONTROL")
    print("================================")
    for item in render_manifest:
        quality_status = str(item.get("quality_status") or item.get("status") or "success").upper()
        rank = item.get("rank", "?")
        if quality_status == "PASS":
            print(f"Clip {rank}: PASS")
        elif quality_status == "WARNING":
            print(f"Clip {rank}: WARNING")
        else:
            print(f"Clip {rank}: FAIL")
    print(f"Passed: {pass_count} | Warnings: {warn_count} | Failed: {fail_count}")

    return render_manifest

