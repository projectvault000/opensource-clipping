import json

from clipping.cache_manager import CacheManager


def test_cache_manager_reuses_json_payloads(tmp_path):
    cache = CacheManager(str(tmp_path))
    payload = {"text": "hello world", "segments": [{"word": "hello"}]}

    assert cache.get_json("transcript", "sample-key") is None

    cache.put_json("transcript", "sample-key", payload)
    cached = cache.get_json("transcript", "sample-key")

    assert cached == payload
    assert cache.get_stats()["transcript"]["hits"] >= 1


def test_cache_manager_tracks_stage_timings(tmp_path):
    cache = CacheManager(str(tmp_path))
    cache.record_timing("transcribe", 0.25)
    cache.record_timing("gemini", 1.5)

    stats = cache.get_stats()
    assert stats["stage_timings"]["transcribe"] >= 0.25
    assert stats["stage_timings"]["gemini"] >= 1.5
