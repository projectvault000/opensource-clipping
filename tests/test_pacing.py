import importlib.util
import sys
import types
from pathlib import Path

import pytest

from clipping.pacing import (
    build_pacing_edit_map,
    map_source_time,
    shift_output_map,
    validate_pacing_map,
)


def _words(*items):
    return [
        {"word": text, "start": start, "end": end}
        for text, start, end in items
    ]


def test_short_natural_pauses_are_preserved():
    pacing = build_pacing_edit_map(
        0.0,
        2.5,
        _words(("Hello,", 0.5, 1.0), ("world.", 1.4, 2.0)),
    )

    assert pacing["removed_duration"] == 0.0
    assert pacing["render_segments"] == [{"start_time": 0.0, "end_time": 2.5}]


def test_long_sentence_pause_is_shortened_and_mapped():
    pacing = build_pacing_edit_map(
        0.0,
        7.4,
        _words(("We.", 1.0, 1.4), ("left.", 4.0, 4.5), ("together.", 6.0, 6.5)),
    )

    assert pacing["shortened_pause_count"] == 1
    assert pacing["removed_duration"] == pytest.approx(1.75)
    assert pacing["output_duration"] == pytest.approx(5.65)
    assert map_source_time(pacing["map"], 6.5) == pytest.approx(4.75)


def test_question_pause_is_preserved_more_than_sentence_pause():
    pacing = build_pacing_edit_map(
        0.0,
        5.4,
        _words(("Did", 0.5, 0.8), ("you", 0.8, 1.0), ("know?", 1.0, 1.3), ("Yes.", 4.6, 5.0)),
    )

    pause = next(item for item in pacing["compressed_pauses"] if item["kind"] == "dramatic")
    assert pause["retained_duration"] == pytest.approx(1.5)
    assert pause["removed_duration"] == pytest.approx(1.8)


def test_mid_sentence_hesitation_is_only_shortened_when_very_long():
    ordinary = build_pacing_edit_map(
        0.0,
        3.5,
        _words(("I", 0.5, 0.7), ("maybe", 2.7, 3.0)),
    )
    long_hesitation = build_pacing_edit_map(
        0.0,
        4.8,
        _words(("I", 0.5, 0.7), ("maybe", 4.0, 4.3)),
    )
def test_pacing_map_rejects_overlapping_or_out_of_range_output():
    assert ordinary["shortened_pause_count"] == 0
    hesitation = next(item for item in long_hesitation["compressed_pauses"] if item["kind"] == "hesitation")
    assert hesitation["retained_duration"] == pytest.approx(1.2)


def test_leading_and_trailing_dead_air_keep_natural_handles():
    pacing = build_pacing_edit_map(
        0.0,
        10.0,
        [{"start": 2.0, "end": 7.0, "text": "A complete explanation."}],
    )

    assert pacing["render_segments"][0]["start_time"] == pytest.approx(1.7)
    assert pacing["render_segments"][-1]["end_time"] == pytest.approx(7.4)
    assert pacing["output_duration"] == pytest.approx(5.7)


def test_overlapping_base_ranges_are_normalized_before_pacing():
    pacing = build_pacing_edit_map(
        0.0,
        10.0,
        [],
        base_segments=[
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 4.5, "end_time": 8.0},
        ],
    )

    assert pacing["render_segments"] == [{"start_time": 0.0, "end_time": 8.0}]
    assert pacing["output_duration"] == 8.0


def test_invalid_word_times_fall_back_to_original_range():
    pacing = build_pacing_edit_map(
        0.0,
        10.0,
        _words(("bad", float("nan"), float("inf"))),
    )

    assert pacing["changed"] is False
    assert pacing["render_segments"] == [{"start_time": 0.0, "end_time": 10.0}]
    assert map_source_time(pacing["map"], 5.0) == 5.0


def test_disabled_pacing_preserves_full_source_range():
    pacing = build_pacing_edit_map(
        0.0,
        10.0,
        _words(("start", 3.0, 3.2)),
        enabled=False,
    )

    assert pacing["removed_duration"] == 0.0
    assert pacing["render_segments"] == [{"start_time": 0.0, "end_time": 10.0}]


def test_final_pacing_map_shifts_after_prefix_and_validates_duration():
    pacing = build_pacing_edit_map(
        0.0,
        8.0,
        _words(("We.", 1.0, 1.4), ("left.", 4.0, 4.5)),
    )
    final_map = shift_output_map(pacing["map"], 4.0)

    assert final_map[0]["output_start"] == 4.0
    assert validate_pacing_map(final_map, 4.0 + pacing["output_duration"])["valid"] is True


def test_pacing_map_rejects_overlapping_or_out_of_range_output():
    invalid_map = [
        {"source_start": 0, "source_end": 2, "output_start": 0, "output_end": 2, "kind": "source"},
        {"source_start": 2, "source_end": 4, "output_start": 1.5, "output_end": 3.5, "kind": "source"},
    ]

    result = validate_pacing_map(invalid_map, 3.0)
    assert result["valid"] is False
    assert result["errors"]


def test_silence_trim_prompt_preserves_meaningful_pauses(monkeypatch):
    for name, attribute in (("yt_dlp", "YoutubeDL"), ("faster_whisper", "WhisperModel")):
        module = types.ModuleType(name)
        setattr(module, attribute, object)
        monkeypatch.setitem(sys.modules, name, module)

    engine_path = Path(__file__).resolve().parents[1] / "clipping" / "engine.py"
    spec = importlib.util.spec_from_file_location("pacing_prompt_engine", engine_path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    prompt = engine.get_analysis_prompt(
        "[1.0 - 2.0] I thought... [4.0 - 5.0] maybe we should leave.",
        1,
        3,
        types.SimpleNamespace(silence_trim=True, no_segment_trim=False, hook_v2=False),
    ).lower()

    assert "jeda emosional/dramatis" in prompt
    assert "jangan membuat micro-cut" in prompt
    assert "jangan sertakan jeda lebih dari 0.5 detik" not in prompt


def test_existing_silence_trim_flags_remain_compatible():
    config_path = Path(__file__).resolve().parents[1] / "clipping" / "config.py"
    spec = importlib.util.spec_from_file_location("pacing_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    args = config._build_parser().parse_args(["--silence-trim", "--no-segment-trim"])

    assert args.silence_trim is True
    assert args.no_segment_trim is True
