import json

from clipping.audio_mastering import (
    TARGET_I,
    TARGET_TP,
    master_audio_in_place,
    parse_loudnorm_json,
)


def test_loudnorm_json_parser_extracts_measurements():
    stderr = """
    [Parsed_loudnorm_0 @ 000] {
        "input_i" : "-19.20",
        "input_tp" : "-3.10",
        "input_lra" : "4.20",
        "input_thresh" : "-29.80",
        "output_i" : "-14.00",
        "target_offset" : "1.20"
    }
    """

    result = parse_loudnorm_json(stderr)

    assert result["input_i"] == "-19.20"
    assert result["target_offset"] == "1.20"


def test_mastering_falls_back_without_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr("shutil.which", lambda name: None)

    result = master_audio_in_place(str(tmp_path / "missing.mp4"))

    assert result["status"] == "skipped"


def test_mastering_targets_documented_loudness_values():
    assert TARGET_I == -14.0
    assert TARGET_TP == -1.0
