from clipping.subtitle_policy import escape_ass_text, group_timed_words, validate_caption_segments


def words(items):
    return [{"word": text, "start": start, "end": end} for text, start, end in items]


def test_groups_at_punctuation_and_pause_without_word_counter_artifacts():
    grouped = group_timed_words(words([
        ("If", 0.0, 0.2),
        ("you", 0.2, 0.4),
        ("look", 0.4, 0.6),
        ("at", 0.6, 0.8),
        ("the", 0.8, 1.0),
        ("numbers,", 1.0, 1.2),
        ("the", 1.8, 2.0),
        ("problem", 2.0, 2.3),
        ("is", 2.3, 2.5),
        ("obvious.", 2.5, 2.8),
    ]), max_words=5)

    assert [item["text"] for item in grouped] == [
        "If you look at the numbers,",
        "the problem is obvious.",
    ]


def test_short_phrase_stays_intact_and_long_phrase_is_bounded():
    short = group_timed_words(words([("Of", 0.0, 0.2), ("course.", 0.2, 0.5)]), max_words=5)
    long = group_timed_words(words([
        ("The", 0.0, 0.2), ("company", 0.2, 0.4), ("was", 0.4, 0.6),
        ("already", 0.6, 0.8), ("running", 0.8, 1.0), ("out", 1.0, 1.2),
        ("of", 1.2, 1.4), ("money.", 1.4, 1.6),
    ]), max_words=5)

    assert [item["text"] for item in short] == ["Of course."]
    assert all(len(item["words"]) <= 6 for item in long)
    assert "money." in long[-1]["text"]


def test_invalid_and_duplicate_word_events_are_repaired():
    grouped = group_timed_words([
        {"word": "the", "start": 0.0, "end": 0.3},
        {"word": "the", "start": 0.2, "end": 0.4},
        {"word": "answer.", "start": 0.4, "end": 0.8},
        {"word": "bad", "start": 2.0, "end": 1.0},
    ], max_words=5)

    assert grouped[0]["text"] == "the answer."
    assert grouped[0]["end"] == 0.8


def test_ass_text_escapes_override_braces_and_newlines():
    assert escape_ass_text("A {danger}\nB") == r"A \{danger\}\NB"

def test_caption_validation_rejects_invalid_overlap_and_duration():
    result = validate_caption_segments([
        {"start": 0.0, "end": 1.0, "text": "first"},
        {"start": 0.8, "end": 3.0, "text": "overlap"},
    ], duration=2.0)

    assert result["valid"] is False
    assert any("overlaps" in error for error in result["errors"])
    assert any("exceeds" in error for error in result["errors"])
