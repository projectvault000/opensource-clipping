import math
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipping.metadata import normalize_hook_plan


_voiceover_spec = importlib.util.spec_from_file_location(
    "hook_test_voiceover",
    Path(__file__).resolve().parents[1] / "clipping" / "voiceover.py",
)
_voiceover = importlib.util.module_from_spec(_voiceover_spec)
_saved_google_modules = {
    name: sys.modules.get(name)
    for name in ("google", "google.genai", "google.genai.types")
}
_google = types.ModuleType("google")
_genai = types.ModuleType("google.genai")
_genai_types = types.ModuleType("google.genai.types")
_genai.types = _genai_types
_google.genai = _genai
sys.modules.update({
    "google": _google,
    "google.genai": _genai,
    "google.genai.types": _genai_types,
})
_voiceover_spec.loader.exec_module(_voiceover)
for _name, _module in _saved_google_modules.items():
    if _module is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module


def _clip(hook_plan=None, **overrides):
    clip = {
        "start_time": 10.0,
        "end_time": 40.0,
        "hook_start_time": 12.0,
        "hook_end_time": 15.0,
    }
    if hook_plan is not None:
        clip["hook_plan"] = hook_plan
    clip.update(overrides)
    return normalize_hook_plan(clip)


def test_hook_plan_accepts_each_supported_strategy():
    for hook_type in ("commentary_hook", "text_hook", "question_hook"):
        plan = _clip({
            "type": hook_type,
            "duration_target": 3.0,
            "text": "Why did the answer change?",
            "source_start": 18.0,
            "source_end": 21.0,
        })
        assert plan["type"] == hook_type
        assert plan["fallback"] is False
        assert (plan["source_start"], plan["source_end"]) == (18.0, 21.0)


def test_hook_plan_accepts_source_teaser_without_generated_text():
    plan = _clip({
        "type": "source_teaser",
        "duration_target": 3.0,
        "text": "",
        "source_start": 18.0,
        "source_end": 21.0,
    })
    assert plan["type"] == "source_teaser"
    assert plan["fallback"] is False


def test_invalid_hook_plan_falls_back_to_legacy_source_interval():
    invalid_plans = [
        {"type": "unsupported", "duration_target": 3, "text": "bad", "source_start": 12, "source_end": 15},
        {"type": "question_hook", "duration_target": 3, "text": "", "source_start": 12, "source_end": 15},
        {"type": "question_hook", "duration_target": 0, "text": "Why?", "source_start": 12, "source_end": 15},
        {"type": "text_hook", "duration_target": 3, "text": "A question", "source_start": -5, "source_end": 999},
        {"type": "source_teaser", "duration_target": math.nan, "text": "", "source_start": 12, "source_end": 15},
        {"type": "source_teaser", "duration_target": 3, "text": "", "source_start": 15, "source_end": 12},
    ]

    for invalid in invalid_plans:
        plan = _clip(invalid)
        assert plan["type"] == "source_teaser"
        assert plan["fallback"] is True
        assert (plan["source_start"], plan["source_end"]) == (12.0, 15.0)


def test_missing_hook_plan_uses_legacy_source_interval():
    plan = _clip()
    assert plan["type"] == "source_teaser"
    assert plan["fallback"] is True
    assert (plan["source_start"], plan["source_end"]) == (12.0, 15.0)


def test_invalid_legacy_interval_falls_back_inside_clip():
    plan = _clip(None, hook_start_time=-2.0, hook_end_time=99.0)
    assert plan["type"] == "source_teaser"
    assert plan["source_start"] == 10.0
    assert plan["source_end"] == 40.0


def test_hook_output_offset_respects_enabled_and_disabled_v1():
    clip = {"hook_plan": {"source_start": 18.0, "source_end": 21.4}}
    enabled = SimpleNamespace(use_hook_glitch=True, durasi_hook=3, hook_v2=False)
    disabled = SimpleNamespace(use_hook_glitch=True, durasi_hook=0, hook_v2=False)

    assert _voiceover.estimate_hook_output_offset(clip, enabled) == pytest.approx(3.4)
    assert _voiceover.estimate_hook_output_offset(clip, enabled, legacy_transition_duration=1.0) == pytest.approx(4.4)
    assert _voiceover.estimate_hook_output_offset(clip, disabled) == 0.0


def test_hook_v2_offset_includes_each_microclip_and_transition():
    clip = {
        "hook_v2": {
            "items": [
                {"start_time": 5.0, "end_time": 6.0},
                {"start_time": 10.0, "end_time": 12.0},
            ]
        }
    }
    cfg = SimpleNamespace(hook_v2=True, white_flash_duration=0.12)

    assert _voiceover.estimate_hook_output_offset(clip, cfg) == pytest.approx(3.24)


def test_clip_analysis_prompt_requires_specific_grounded_hook_plan(monkeypatch):
    for name, attribute in (("yt_dlp", "YoutubeDL"), ("faster_whisper", "WhisperModel")):
        module = types.ModuleType(name)
        setattr(module, attribute, object)
        monkeypatch.setitem(sys.modules, name, module)

    engine_path = Path(__file__).resolve().parents[1] / "clipping" / "engine.py"
    spec = importlib.util.spec_from_file_location("hook_test_engine", engine_path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    prompt = engine.get_analysis_prompt(
        "[10.0 - 12.0] His answer changed.",
        1,
        3,
        SimpleNamespace(hook_v2=False, no_segment_trim=True, voiceover=True),
    )

    assert '"hook_plan"' in prompt
    assert "source_teaser" in prompt
    assert "commentary_hook" in prompt
    assert "question_hook" in prompt
    assert "payoff" in prompt.lower()
    assert "jangan menulis ulang dialog seolah-olah diucapkan" in prompt
    assert "you won't believe" in prompt.lower()


def test_cli_preserves_zero_hook_duration():
    config_path = Path(__file__).resolve().parents[1] / "clipping" / "config.py"
    spec = importlib.util.spec_from_file_location("hook_test_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    args = config._build_parser().parse_args(["--hook-duration", "0"])
    assert args.hook_duration == 0
