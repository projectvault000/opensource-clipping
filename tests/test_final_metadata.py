import json

from clipping.final_metadata import (
    build_final_edit_context,
    choose_thumbnail_text,
    choose_title,
    sanitize_filename,
    validate_metadata_package,
    write_metadata_sidecar,
)


def test_title_fallback_rejects_generic_clickbait():
    item = {
        "title_inggris": "You Won't Believe What Happened Next!",
        "description_hook": "Why the answer changed after one question.",
    }

    assert choose_title(item) == "Why the answer changed after one question"


def test_thumbnail_text_is_short_and_not_full_title():
    item = {
        "title_inggris": "Why His Answer Changed After One Question",
        "hook_plan": {"type": "question_hook", "text": "Why did he change his answer?"},
    }

    assert choose_thumbnail_text(item) == "WHY DID HE CHANGE HIS ANSWER"
    assert len(choose_thumbnail_text(item).split()) <= 6


def test_final_context_contains_commentary_and_edit_signals():
    context = build_final_edit_context({
        "start_time": 10.0,
        "end_time": 40.0,
        "hook_plan": {"type": "source_teaser", "source_start": 12.0, "source_end": 15.0},
        "pacing_removed_duration": 1.4,
        "voiceover": {
            "plan": {"segments": [
                {"type": "commentary", "narration": "This detail changes the context."},
            ]}
        },
        "broll_assets": [{"asset_id": "123"}],
        "quality_status": "pass",
    })

    assert context["commentary"] == ["This detail changes the context."]
    assert context["pacing_removed_duration"] == 1.4
    assert context["quality_status"] == "pass"


def test_metadata_sidecar_is_structured_and_filename_safe(tmp_path):
    item = {
        "title_inggris": "Why His Answer Changed?",
        "description_context": "The clip follows the question and the response.",
        "hastag": "#interviews #communication",
        "youtube_tags_final": ["interviews", "communication"],
        "thumbnail_source_time": 14.2,
    }
    output = tmp_path / "clip_metadata.json"

    payload = write_metadata_sidecar(item, str(output))

    assert sanitize_filename('Why: His Answer / Changed?') == "Why-His-Answer-Changed"
    assert json.loads(output.read_text(encoding="utf-8"))["thumbnail_source_time"] == 14.2
    assert payload["title"] == "Why His Answer Changed?"
    assert payload["warnings"] == []
