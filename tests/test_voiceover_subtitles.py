import importlib.util
import sys
import types

# Stub the optional external modules that are not available in this minimal environment.
google = types.ModuleType("google")
genai = types.ModuleType("google.genai")
genai_types = types.ModuleType("google.genai.types")
genai.types = genai_types
google.genai = genai
sys.modules["google"] = google
sys.modules["google.genai"] = genai
sys.modules["google.genai.types"] = genai_types

spec = importlib.util.spec_from_file_location(
    "voiceover_mod",
    "clipping/voiceover.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_consolidate_segments_respects_requested_group_size():
    raw = [
        {"start": 0.0, "end": 0.5, "text": "A", "words": [{"word": "A", "start": 0.0, "end": 0.5}]},
        {"start": 0.5, "end": 1.0, "text": "B", "words": [{"word": "B", "start": 0.5, "end": 1.0}]},
        {"start": 1.0, "end": 1.5, "text": "C", "words": [{"word": "C", "start": 1.0, "end": 1.5}]},
        {"start": 1.5, "end": 2.0, "text": "D", "words": [{"word": "D", "start": 1.5, "end": 2.0}]},
    ]

    segments = mod._consolidate_segments(raw, words_per_seg=2)

    assert [s["text"] for s in segments] == ["A B", "C D"]
    assert all(s["end"] >= s["start"] for s in segments)


def test_synthesize_voice_accepts_word_budget_and_returns_text_chunks():
    result = mod.synthesize_voice(
        "alpha beta gamma delta",
        "en-US-AvaNeural",
        ".",
        "clip_1",
        max_words_per_subtitle=2,
    )

    assert len(result) == 2
    assert result[1][0]["text"].startswith("gamma")
