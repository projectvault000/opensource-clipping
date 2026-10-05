import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import types


google = types.ModuleType("google")
genai = types.ModuleType("google.genai")
genai_types = types.ModuleType("google.genai.types")
genai.types = genai_types
google.genai = genai
sys.modules["google"] = google
sys.modules["google.genai"] = genai
sys.modules["google.genai.types"] = genai_types

spec = importlib.util.spec_from_file_location("voiceover_mod", "clipping/voiceover.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_commentary_plan_validation_rejects_bad_ranges():
    plan = {
        "segments": [
            {"type": "source", "source_start": 0.0, "source_end": 10.0},
            {"type": "commentary", "insert_after_source_time": 5.0, "narration": "This matters."},
            {"type": "source", "source_start": 12.0, "source_end": 8.0},
        ]
    }

    result = mod.validate_commentary_plan(plan, clip_duration=20.0)
    assert result["valid"] is False
    assert "source_end" in result["errors"][0]


def test_commentary_plan_timeline_maps_source_to_output_offsets():
    plan = {
        "segments": [
            {"type": "source", "source_start": 0.0, "source_end": 8.0},
            {"type": "commentary", "insert_after_source_time": 8.0, "narration": "Here is the context."},
            {"type": "source", "source_start": 8.0, "source_end": 16.0},
        ]
    }

    timeline = mod.build_commentary_timeline(plan, clip_duration=16.0)
    assert timeline[0]["type"] == "source"
    assert timeline[1]["type"] == "commentary"
    assert timeline[1]["output_start"] == 8.0
    assert timeline[2]["source_start"] == 8.0
    assert timeline[2]["output_start"] == 8.0 + 5.0


def test_commentary_timeline_accumulates_hook_and_multiple_insertions():
    plan = {
        "segments": [
            {"type": "source", "source_start": 0.0, "source_end": 8.0},
            {"type": "commentary", "insert_after_source_time": 8.0, "narration": "First intervention."},
            {"type": "source", "source_start": 8.0, "source_end": 12.0},
            {"type": "commentary", "insert_after_source_time": 12.0, "narration": "Second intervention."},
            {"type": "source", "source_start": 12.0, "source_end": 16.0},
        ]
    }

    timeline = mod.build_commentary_timeline(
        plan,
        clip_duration=16.0,
        initial_output_offset=3.0,
    )

    assert timeline[0]["output_start"] == 3.0
    assert timeline[0]["output_end"] == 11.0
    assert timeline[2]["output_start"] == 16.0
    assert timeline[2]["output_end"] == 20.0
    assert timeline[4]["output_start"] == 25.0
    assert timeline[4]["output_end"] == 29.0


def test_commentary_audio_filter_contains_ducking_and_fades():
    mix = mod.build_commentary_audio_filter(
        source_audio_label="[1:a]",
        narration_audio_label="[2:a]",
        original_volume=0.15,
        voiceover_volume=1.0,
        narration_duration=12.0,
    )

    assert "afade=t=in" in mix
    assert "afade=t=out" in mix
    assert "volume=0.15" in mix
    assert "amix" in mix


def test_commentary_visual_validation_accepts_supported_modes_and_rejects_invalid():
    valid = mod.validate_commentary_visual({"mode": "replay", "source_start": 2.0, "source_end": 6.0})
    assert valid["mode"] == "replay"
    assert valid["source_start"] == 2.0

    invalid = mod.validate_commentary_visual({"mode": "explosion_effect", "source_start": -1, "source_end": 15})
    assert invalid["mode"] == "continue"
    assert invalid["fallback"] is True


def test_commentary_plan_visual_falls_back_to_continue_when_replay_invalid():
    plan = {
        "segments": [
            {"type": "source", "source_start": 0.0, "source_end": 10.0},
            {"type": "commentary", "insert_after_source_time": 5.0, "narration": "This matters.", "visual": {"mode": "replay", "source_start": 20.0, "source_end": 12.0}},
            {"type": "source", "source_start": 10.0, "source_end": 20.0},
        ]
    }

    fixed = mod.validate_commentary_plan(plan, clip_duration=20.0)
    assert fixed["valid"] is True
    assert fixed["segments"][1]["visual"]["mode"] == "continue"


def test_commentary_broll_requires_specific_safe_query_and_intent():
    invalid = mod.validate_commentary_visual({
        "mode": "broll",
        "query": "business",
        "visual_intent": "A stock illustration",
    })
    valid = mod.validate_commentary_visual({
        "mode": "broll",
        "query": "semiconductor clean room manufacturing",
        "visual_intent": "Illustrate a controlled chip manufacturing environment",
    })

    assert invalid["mode"] == "continue"
    assert invalid["fallback"] is True
    assert valid["mode"] == "broll"
    assert valid["visual_intent"]


def test_single_broll_visual_is_not_applied_to_multiple_commentary_sections():
    plan = {
        "segments": [
            {
                "type": "commentary",
                "narration": "A concept needs illustration.",
                "visual": {
                    "mode": "broll",
                    "query": "semiconductor clean room manufacturing",
                    "visual_intent": "Illustrate a controlled chip manufacturing environment",
                },
            },
            {"type": "commentary", "narration": "Now watch the speaker respond.", "visual": {"mode": "continue"}},
        ]
    }

    visual = mod.resolve_commentary_visual(plan, source_duration=30.0)
    assert visual["mode"] == "continue"


def test_commentary_prompts_require_value_and_grounding():
    prompt = mod.get_commentary_timeline_prompt(
        "I sold the company for ten million dollars.",
        "reaction",
        "en",
        "short",
        clip_duration=18.0,
    )

    lowered = prompt.lower()
    assert "add information" in lowered or "add value" in lowered
    assert "do not merely restate" in lowered or "merely restate" in lowered
    assert "grounded" in lowered or "supported by the source" in lowered
    assert "payoff" in lowered or "spoiler" in lowered


def test_validate_final_output_rejects_missing_file():
    missing = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    missing.close()
    missing_path = missing.name
    import os
    os.unlink(missing_path)

    result = mod.validate_final_output(missing_path, expected_duration=5.0, expected_ratio="9:16")
    assert result["status"] == "fail"
    assert result["errors"]


def test_validate_final_output_accepts_valid_generated_render(tmp_path):
    if shutil.which("ffmpeg") is None:
        return

    out_path = tmp_path / "valid_commentary.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1080x1920:rate=30:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    result = mod.validate_final_output(str(out_path), expected_duration=1.0, expected_ratio="9:16")
    assert result["status"] in {"pass", "warning"}
    assert result["checks"].get("file_exists") == "pass"
