import json
import zipfile

from clipping.export_package import build_final_srt, export_publishing_package


def test_final_srt_uses_prefix_and_pacing_map():
    row = {
        "hook_output_offset": 3.0,
        "vo_render_duration": 2.0,
        "start_time": 10.0,
        "pacing_map": [
            {"source_start": 10.0, "source_end": 20.0, "output_start": 0.0, "output_end": 8.0}
        ],
    }
    srt = build_final_srt(row, [{"start": 12.0, "end": 14.0, "text": "Source caption"}])

    assert "00:00:06,600 --> 00:00:08,200" in srt
    assert "Source caption" in srt


def test_final_srt_does_not_double_shift_final_pacing_map():
    row = {
        "hook_output_offset": 3.0,
        "vo_render_duration": 2.0,
        "start_time": 10.0,
        "pacing_final_map": [
            {"source_start": 10.0, "source_end": 20.0, "output_start": 5.0, "output_end": 13.0}
        ],
    }
    srt = build_final_srt(row, [{"start": 12.0, "end": 14.0, "text": "Source caption"}])

    assert "00:00:06,600 --> 00:00:08,200" in srt


def test_export_creates_allowlisted_package_and_zip(tmp_path):
    video = tmp_path / "final.mp4"
    thumb = tmp_path / "thumb.jpg"
    video.write_bytes(b"final-video")
    thumb.write_bytes(b"thumbnail")
    manifest = [{
        "rank": 1,
        "status": "success",
        "quality_status": "pass",
        "video_path": str(video),
        "thumbnail_path": str(thumb),
        "title_inggris": "Why His Answer Changed",
        "youtube_title_final": "Why His Answer Changed",
        "youtube_description_final": "A concise description.",
        "hastag": "#interviews #communication",
        "youtube_tags_final": ["interviews", "communication"],
        "start_time": 10.0,
        "end_time": 30.0,
        "quality_control": {"status": "pass", "checks": {"file_exists": "pass"}},
    }]

    summary = export_publishing_package(manifest, str(tmp_path))
    package = tmp_path / "publishing_package" / "clip_01"
    expected = {"video.mp4", "thumbnail.jpg", "metadata.json", "publish.txt", "qc_report.json", "edit_plan.json", "source_info.json", "manifest.json"}

    assert summary["pass"] == 1
    assert expected.issubset({path.name for path in package.iterdir()})
    assert (tmp_path / "publishing_package.zip").exists()
    with zipfile.ZipFile(tmp_path / "publishing_package.zip") as archive:
        names = set(archive.namelist())
    assert "publishing_package/clip_01/video.mp4" in names
    assert "publishing_package/clip_01/metadata.json" in names
    assert "publishing_package/clip_01/publish.txt" in names


def test_failed_clip_is_listed_but_not_marked_pass(tmp_path):
    summary = export_publishing_package([{
        "rank": 2,
        "quality_status": "fail",
        "quality_control": {"status": "fail", "errors": ["bad media"]},
        "title_inggris": "Broken clip",
    }], str(tmp_path))

    assert summary["fail"] == 1
    assert summary["clips"][0]["status"] == "FAIL"
