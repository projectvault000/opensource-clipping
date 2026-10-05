from clipping.visual_plan import compute_voiceover_visual_plan


def test_longer_narration_requires_looped_source_visual():
    plan = compute_voiceover_visual_plan(7.8, 12.3)

    assert plan["source_duration"] == 7.8
    assert plan["narration_duration"] == 12.3
    assert plan["loop_required"] is True
    assert plan["target_duration"] >= 12.3
    assert plan["loop_count"] >= 2


def test_shorter_narration_does_not_force_looping():
    plan = compute_voiceover_visual_plan(12.5, 7.8)

    assert plan["loop_required"] is False
    assert plan["target_duration"] >= 7.8
    assert plan["loop_count"] == 1
