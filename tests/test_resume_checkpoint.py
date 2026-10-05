import json
import os
from types import SimpleNamespace

from clipping.checkpoints import (
    JobCheckpoint, deterministic_job_id, job_id_for_config,
    prepare_cli_job, render_manifest_complete,
)


def test_deterministic_job_id_is_stable(tmp_path):
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    cfg = {"url_youtube": url, "clips": 5, "ratio": "9:16"}

    first = deterministic_job_id(url, cfg)
    second = deterministic_job_id(url, cfg)
    third = deterministic_job_id(url, {"clips": 5, "ratio": "9:16", "url_youtube": url})

    assert first == second == third
    assert len(first) > 8


def test_checkpoint_records_stage_progress(tmp_path):
    job_path = tmp_path / "resume-job"
    cp = JobCheckpoint(job_path)

    assert cp.status() == "new"
    cp.mark_started("download")
    cp.mark_succeeded("download", artifact_path="video_asli.mp4")

    assert cp.is_complete("download") is True
    assert cp.status() == "running" or cp.status() == "complete"

    state = cp.load()
    assert state["stages"]["download"]["status"] == "succeeded"
    assert state["stages"]["download"]["artifact_path"] == "video_asli.mp4"

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([{"rank": 1}]), encoding="utf-8")
    cp.mark_succeeded("render", manifest_path=str(manifest_path))
    assert cp.stage_artifact("render") == str(manifest_path)


def test_cli_job_directory_is_stable_and_separate_from_secrets(tmp_path):
    cfg = SimpleNamespace(
        base_dir=str(tmp_path), outputs_dir=str(tmp_path / "outputs"),
        file_video_asli=str(tmp_path / "video_asli.mp4"),
        url_youtube="https://example.com/video", jumlah_clip=5,
        api_key_gemini="first-secret", pexels_api_key="second-secret",
    )
    job_id = job_id_for_config(cfg)
    cfg.api_key_gemini = "rotated-secret"
    assert job_id_for_config(cfg) == job_id
    prepare_cli_job(cfg)
    assert cfg.outputs_dir == str(tmp_path / "outputs" / job_id)
    assert cfg.file_video_asli == str(tmp_path / "outputs" / job_id / "video_asli.mp4")
    cfg.jumlah_clip = 1
    cfg.job_id = None
    assert job_id_for_config(cfg) != job_id


def test_render_resume_requires_every_video(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"media")
    row = {"status": "success", "video_path": str(video)}
    assert render_manifest_complete([row], 1)
    assert not render_manifest_complete([row], 2)
    video.unlink()
    assert not render_manifest_complete([row], 1)
