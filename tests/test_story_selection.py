import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from clipping.metadata import normalize_and_validate, validate_story_candidates


def _candidate(start, end, score, core_start=None, core_end=None, reason="complete moment"):
    return {
        "start_time": start,
        "end_time": end,
        "viral_score": score,
        "core_start_time": start if core_start is None else core_start,
        "core_end_time": end if core_end is None else core_end,
        "story_structure": {"setup": True, "development": True, "payoff": True},
        "alasan": reason,
    }


def test_story_candidates_keep_core_and_story_structure():
    candidate = _candidate(10.0, 40.0, 90, 18.0, 31.0)

    result = validate_story_candidates([candidate], source_duration=100.0)

    assert len(result) == 1
    assert result[0]["core_start_time"] == 18.0
    assert result[0]["core_end_time"] == 31.0
    assert result[0]["story_structure"] == {"setup": True, "development": True, "payoff": True}


def test_story_candidates_reject_invalid_clip_ranges():
    candidates = [
        _candidate(-1.0, 30.0, 99),
        _candidate(30.0, 20.0, 98),
        _candidate(10.0, 29.0, 97),
        _candidate(10.0, 190.0, 96),
        _candidate(90.0, 110.0, 95),
        _candidate(float("nan"), 30.0, 94),
    ]

    assert validate_story_candidates(candidates, source_duration=100.0) == []


def test_story_candidates_repair_invalid_core_to_full_selection():
    candidate = _candidate(10.0, 40.0, 90, -2.0, 500.0)

    result = validate_story_candidates([candidate], source_duration=100.0)

    assert result[0]["core_start_time"] == 10.0
    assert result[0]["core_end_time"] == 40.0


def test_story_candidates_drop_heavy_overlap_when_core_is_repeated():
    first = _candidate(0.0, 40.0, 95, 12.0, 22.0)
    duplicate = _candidate(4.0, 44.0, 90, 13.0, 21.0)
    distinct = _candidate(45.0, 80.0, 85, 55.0, 68.0)

    result = validate_story_candidates([duplicate, distinct, first], source_duration=100.0)

    assert [item["viral_score"] for item in result] == [95, 85]


def test_story_candidates_keep_overlapping_ranges_with_distinct_cores():
    first = _candidate(0.0, 40.0, 95, 2.0, 8.0)
    distinct = _candidate(4.0, 44.0, 90, 31.0, 38.0)

    result = validate_story_candidates([first, distinct], source_duration=100.0)

    assert len(result) == 2


def test_metadata_normalization_keeps_valid_story_candidate_fields():
    candidate = _candidate(10.0, 40.0, 90, 18.0, 31.0)
    candidate.update({
        "rank": 4,
        "title_indonesia": "Konteks dan konsekuensi",
        "title_inggris": "Context and consequence",
        "description_hook": "A concise hook.",
        "description_context": "The clip explains the decision and its result.",
        "hastag": "#story #decision",
        "keyword_tags": ["story", "decision", "context", "result", "interview"],
        "tiktok_title_id": "Konteks keputusan dan dampaknya",
        "tiktok_caption_id": "Cerita lengkap tentang keputusan dan dampaknya.",
        "tiktok_caption": "A complete story about a decision and its impact.",
    })

    result = normalize_and_validate([candidate], source_duration=100.0)

    assert len(result) == 1
    assert result[0]["rank"] == 1
    assert result[0]["start_time"] == 10.0
    assert result[0]["core_start_time"] == 18.0
    assert result[0]["story_structure"]["payoff"] is True


def test_selection_prompt_prioritizes_complete_moments_and_natural_boundaries(monkeypatch):
    for name, attribute in (("yt_dlp", "YoutubeDL"), ("faster_whisper", "WhisperModel")):
        module = types.ModuleType(name)
        setattr(module, attribute, object)
        monkeypatch.setitem(sys.modules, name, module)

    engine_path = Path(__file__).resolve().parents[1] / "clipping" / "engine.py"
    spec = importlib.util.spec_from_file_location("story_test_engine", engine_path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    transcript = "[10.0 - 12.0] We built the product.\n[12.1 - 15.0] Customers left, so we shut it down."
    prompt = engine.get_analysis_prompt(transcript, 1, 3)
    lowered = prompt.lower()

    assert "core_start_time" in prompt
    assert "core_end_time" in prompt
    assert "story_structure" in prompt
    assert "setup" in lowered and "payoff" in lowered
    assert "jangan mulai di tengah kalimat" in lowered
    assert "commentary" in lowered
    assert transcript in prompt


def test_gemini_schema_requires_core_range_and_story_structure(monkeypatch):
    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured["config"] = kwargs["config"]
            return SimpleNamespace(text="[]")

    class FakeClient:
        models = FakeModels()

    fake_types = types.ModuleType("google.genai.types")
    fake_types.HttpOptions = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_types.HttpRetryOptions = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_types.GenerateContentConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_genai = types.ModuleType("google.genai")
    fake_genai.types = fake_types
    fake_genai.Client = lambda **kwargs: FakeClient()
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    for name, attribute in (("yt_dlp", "YoutubeDL"), ("faster_whisper", "WhisperModel")):
        module = types.ModuleType(name)
        setattr(module, attribute, object)
        monkeypatch.setitem(sys.modules, name, module)

    engine_path = Path(__file__).resolve().parents[1] / "clipping" / "engine.py"
    spec = importlib.util.spec_from_file_location("story_schema_engine", engine_path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.analyze_with_gemini(
        "[10.0 - 12.0] The complete source moment.",
        SimpleNamespace(
            api_key_gemini="test-key",
            jumlah_clip=1,
            durasi_hook=3,
            gemini_model="test-model",
            gemini_fallback_model=None,
            hook_v2=False,
            no_segment_trim=True,
        ),
    )

    properties = captured["config"].response_schema["items"]["properties"]
    assert "core_start_time" in properties
    assert "core_end_time" in properties
    assert properties["story_structure"]["required"] == ["setup", "development", "payoff"]
    assert "visual_intent" in properties["broll_list"]["items"]["properties"]


def test_nvidia_schema_requires_core_range_and_story_structure(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["schema"] = kwargs["extra_body"]["nvext"]["guided_json"]
            message = SimpleNamespace(content="[]")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    openai = types.ModuleType("openai")
    openai.OpenAI = lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setitem(sys.modules, "openai", openai)
    for name, attribute in (("yt_dlp", "YoutubeDL"), ("faster_whisper", "WhisperModel")):
        module = types.ModuleType(name)
        setattr(module, attribute, object)
        monkeypatch.setitem(sys.modules, name, module)

    engine_path = Path(__file__).resolve().parents[1] / "clipping" / "engine.py"
    spec = importlib.util.spec_from_file_location("story_nvidia_schema_engine", engine_path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.analyze_with_nvidia(
        "[10.0 - 12.0] The complete source moment.",
        SimpleNamespace(
            api_key_nvidia="test-key",
            nvidia_model="test-model",
            jumlah_clip=1,
            durasi_hook=3,
            hook_v2=False,
            no_segment_trim=True,
        ),
    )

    properties = captured["schema"]["items"]["properties"]
    assert "core_start_time" in properties
    assert "core_end_time" in properties
    assert properties["story_structure"]["required"] == ["setup", "development", "payoff"]
    assert "visual_intent" in properties["broll_list"]["items"]["properties"]
