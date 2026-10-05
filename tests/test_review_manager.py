import json
import zipfile
import pytest

from clipping.review_manager import ReviewManager, build_approved_manifest


def test_review_manager_persists_review_state_and_detects_stale_approval(tmp_path):
    job_dir = tmp_path / "job_demo"
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest = [
        {"rank": 1, "clip_id": "clip_01", "quality_status": "pass", "status": "success", "video_path": "a.mp4", "thumbnail_path": "t.jpg", "title_inggris": "Alpha"},
        {"rank": 2, "clip_id": "clip_02", "quality_status": "warning", "status": "success", "video_path": "b.mp4", "thumbnail_path": "t2.jpg", "title_inggris": "Bravo"},
    ]
    manifest_path = job_dir / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    manager = ReviewManager(job_dir)
    manager.sync_manifest(manifest)
    manager.set_review_status("clip_01", "APPROVED", note="Looks good")

    clipped = manager.get_clip("clip_01")
    assert clipped["review_status"] == "APPROVED"
    assert clipped["review_note"] == "Looks good"
    assert manager.get_review_summary()["approved"] == 1

    manifest[0]["title_inggris"] = "Alpha v2"
    manager.sync_manifest(manifest)
    stale = manager.get_clip("clip_01")
    assert stale["review_status"] == "STALE"


def test_build_approved_manifest_excludes_failed_and_unreviewed_clips(tmp_path):
    manifest = [
        {"rank": 1, "clip_id": "clip_01", "quality_status": "pass", "status": "success", "review_status": "APPROVED", "package_path": "/tmp/clip_01"},
        {"rank": 2, "clip_id": "clip_02", "quality_status": "pass", "status": "success", "review_status": "PENDING", "package_path": "/tmp/clip_02"},
        {"rank": 3, "clip_id": "clip_03", "quality_status": "fail", "status": "failed", "review_status": "PENDING", "package_path": "/tmp/clip_03"},
        {"rank": 4, "clip_id": "clip_04", "quality_status": "warning", "status": "success", "review_status": "REJECTED", "package_path": "/tmp/clip_04"},
    ]

    approved = build_approved_manifest(manifest)
    assert [item["clip_id"] for item in approved] == ["clip_01"]


def test_approved_media_change_becomes_stale(tmp_path):
    job_dir = tmp_path / "job_demo"
    job_dir.mkdir()
    video = job_dir / "clip.mp4"
    video.write_bytes(b"first")
    row = {"clip_id": "clip_01", "status": "success", "quality_status": "pass", "video_path": str(video)}
    manager = ReviewManager(job_dir)
    manager.sync_manifest([row])
    manager.set_review_status("clip_01", "APPROVED")
    video.write_bytes(b"changed media")
    manager.sync_manifest([row])
    assert manager.get_clip("clip_01")["review_status"] == "STALE"
    with pytest.raises(ValueError, match="No approved clips"):
        manager.export_approved_set()


def test_approved_export_contains_only_current_job_files(tmp_path):
    job_dir = tmp_path / "job_demo"
    package = job_dir / "publishing_package" / "clip_01"
    package.mkdir(parents=True)
    (package / "video.mp4").write_bytes(b"media")
    row = {
        "clip_id": "clip_01", "status": "success", "quality_status": "pass",
        "publishing_package_path": str(package),
    }
    manager = ReviewManager(job_dir)
    manager.sync_manifest([row])
    manager.set_review_status("clip_01", "APPROVED")
    result = manager.export_approved_set()
    with zipfile.ZipFile(result["zip_path"]) as archive:
        assert archive.namelist() == [
            "approved_manifest.json", "publishing_package/clip_01/video.mp4"
        ]


def test_approved_export_rejects_package_outside_job(tmp_path):
    job_dir = tmp_path / "job_demo"
    job_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("private", encoding="utf-8")
    row = {
        "clip_id": "clip_01", "status": "success", "quality_status": "pass",
        "publishing_package_path": str(outside),
    }
    manager = ReviewManager(job_dir)
    manager.sync_manifest([row])
    manager.set_review_status("clip_01", "APPROVED")
    with pytest.raises(ValueError, match="outside the job directory"):
        manager.export_approved_set()
