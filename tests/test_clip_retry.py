import json
from types import SimpleNamespace

import pytest

from clipping.export_package import export_publishing_package
from clipping.retry import load_retry_config, normalize_retry_plan, save_run_config, validate_ids
from clipping.review_manager import ReviewManager


def test_retry_config_omits_secrets_and_restores_job(tmp_path, monkeypatch):
    job_id = "job_0123456789abcdef"
    job_dir = tmp_path / job_id
    cfg = SimpleNamespace(
        job_id=job_id, outputs_dir=str(job_dir),
        file_video_asli=str(job_dir / "video_asli.mp4"),
        url_youtube="https://example.com/video", jumlah_clip=2,
        api_key_gemini="secret", hf_token="private", pexels_api_key="key",
        yt_cookies="cookie-file",
    )
    save_run_config(cfg)
    saved = json.loads((job_dir / "run_config.json").read_text(encoding="utf-8"))
    assert not {"api_key_gemini", "hf_token", "pexels_api_key", "yt_cookies"} & saved.keys()
    monkeypatch.setenv("GOOGLE_API_KEY", "fresh-key")
    restored, rank = load_retry_config(tmp_path, job_id, "clip_02")
    assert rank == 2
    assert restored.api_key_gemini == "fresh-key"
    assert restored.outputs_dir == str(job_dir)
    assert restored.yt_cookies is None
    with pytest.raises(ValueError, match="Invalid job ID"):
        validate_ids("../outside", "clip_02")


def test_new_generation_resets_only_target_review(tmp_path):
    manager = ReviewManager(tmp_path / "job")
    rows = [
        {"rank": 1, "status": "success", "quality_status": "pass", "video_path": "first.mp4"},
        {"rank": 2, "status": "success", "quality_status": "pass", "video_path": "second.mp4"},
    ]
    manager.sync_manifest(rows)
    manager.set_review_status("clip_01", "APPROVED")
    manager.set_review_status("clip_02", "NEEDS_CHANGES", note="Fix caption")
    rows[1] = {**rows[1], "generation_version": 2, "video_path": "second-v2.mp4"}
    manager.sync_manifest(rows)
    assert manager.get_clip("clip_01")["review_status"] == "APPROVED"
    retried = manager.get_clip("clip_02")
    assert retried["generation_version"] == 2
    assert retried["review_status"] == "PENDING"
    assert retried["review_note"] == ""


def test_retry_plan_preserves_original_rank_after_normalization():
    plan = [
        {"rank": 1, "start_time": 0, "end_time": 30, "viral_score": 10},
        {"rank": 2, "start_time": 40, "end_time": 70, "viral_score": 9},
    ]
    selected = normalize_retry_plan(plan, 2, 100, 20, 179)
    assert len(selected) == 1
    assert selected[0]["rank"] == 2
    assert selected[0]["start_time"] == 40
    with pytest.raises(ValueError, match="exactly one"):
        normalize_retry_plan(plan, 3, 100, 20, 179)


def test_targeted_package_keeps_other_clip_files(tmp_path, monkeypatch):
    from clipping import export_package

    root = tmp_path / "job"
    first = root / "publishing_package" / "clip_01"
    second = root / "publishing_package" / "clip_02"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "video.mp4").write_bytes(b"untouched")
    (first / "manifest.json").write_text(json.dumps({"clip_id": "clip_01", "status": "PASS"}), encoding="utf-8")
    (second / "old.txt").write_text("old", encoding="utf-8")

    def package_target(row, package_root, source_segments=None):
        target = package_root / "clip_02"
        target.mkdir()
        (target / "video.mp4").write_bytes(b"new")
        return {"id": "clip_02", "status": "PASS", "title": "New", "path": str(target)}

    monkeypatch.setattr(export_package, "_package_clip", package_target)
    summary = export_publishing_package(
        [{"rank": 1, "title_inggris": "First"}, {"rank": 2, "title_inggris": "New"}],
        root, target_rank=2,
    )
    assert summary["pass"] == 2
    assert (first / "video.mp4").read_bytes() == b"untouched"
    assert not (second / "old.txt").exists()
    assert (second / "video.mp4").read_bytes() == b"new"
