from clipping.broll_policy import (
    get_asset_metadata,
    get_used_asset_ids,
    mark_asset_used,
    normalize_broll_query,
    normalize_visual_intent,
    rank_pexels_candidates,
    record_asset_metadata,
    reset_broll_session,
    simplify_broll_query,
)


def _video(asset_id, duration, width, height, link=None):
    return {
        "id": asset_id,
        "duration": duration,
        "url": f"https://www.pexels.com/video/{asset_id}/",
        "video_files": [
            {
                "file_type": "video/mp4",
                "quality": "hd",
                "width": width,
                "height": height,
                "link": link or f"https://videos.pexels.com/{asset_id}.mp4",
            }
        ],
    }


def test_broll_query_accepts_specific_visual_and_rejects_generic_or_unsafe():
    assert normalize_broll_query("  semiconductor   clean room manufacturing ") == "semiconductor clean room manufacturing"
    assert normalize_broll_query("business") is None
    assert normalize_broll_query("https://example.com/clip") is None
    assert normalize_broll_query("ffmpeg -i source.mp4") is None
    assert normalize_broll_query("office success money people working teamwork meeting camera laptop brand") is None
    assert normalize_broll_query("first line\nsecond line") is None
    assert simplify_broll_query("semiconductor wafer fabrication clean room manufacturing") == "semiconductor wafer fabrication clean"


def test_broll_visual_intent_requires_concrete_illustrative_purpose():
    assert normalize_visual_intent("Illustrate a controlled chip manufacturing environment")
    assert normalize_visual_intent(" ") is None
    assert normalize_visual_intent("x" * 300) is None


def test_pexels_ranking_filters_short_low_resolution_and_unusable_assets():
    candidates = rank_pexels_candidates(
        [
            _video("short", 3.0, 1080, 1920),
            _video("small", 9.0, 360, 640),
            _video("wide", 8.0, 1920, 1080),
            _video("portrait", 7.0, 1080, 1920),
            {"id": "no-mp4", "duration": 8.0, "video_files": []},
        ],
        target_ratio=9 / 16,
        minimum_duration=5.0,
    )

    assert [item["asset_id"] for item in candidates] == ["portrait", "wide"]
    assert candidates[0]["file_url"].endswith("portrait.mp4")


def test_candidate_ranking_excludes_already_used_asset_ids():
    candidates = rank_pexels_candidates(
        [_video("used", 8.0, 1080, 1920), _video("fresh", 8.0, 1080, 1920)],
        target_ratio=9 / 16,
        minimum_duration=5.0,
        used_ids={"used"},
    )

    assert [item["asset_id"] for item in candidates] == ["fresh"]


def test_broll_session_state_and_asset_metadata_are_scoped_and_resettable(tmp_path):
    first_job = str(tmp_path / "job-a")
    second_job = str(tmp_path / "job-b")
    reset_broll_session(first_job)
    reset_broll_session(second_job)
    mark_asset_used(first_job, "asset-a")
    record_asset_metadata(first_job, "output-a.mp4", {"asset_id": "asset-a", "query": "clean room"})

    assert get_used_asset_ids(first_job) == {"asset-a"}
    assert get_used_asset_ids(second_job) == set()
    assert get_asset_metadata(first_job, "output-a.mp4")["asset_id"] == "asset-a"

    reset_broll_session(first_job)
    assert get_used_asset_ids(first_job) == set()
    assert get_asset_metadata(first_job, "output-a.mp4") is None
