"""Duration planning for source visuals under generated narration."""

import math


def compute_voiceover_visual_plan(source_duration: float, narration_duration: float):
    source_duration = max(float(source_duration), 0.0)
    narration_duration = max(float(narration_duration), 0.0)

    loop_required = narration_duration > source_duration + 1e-6
    loop_count = 1
    target_duration = source_duration

    if loop_required:
        loop_count = max(1, int(math.ceil(narration_duration / source_duration))) if source_duration > 0 else 1
        target_duration = max(narration_duration, source_duration)
    elif source_duration <= 0:
        target_duration = narration_duration

    return {
        "source_duration": source_duration,
        "narration_duration": narration_duration,
        "loop_required": loop_required,
        "loop_count": loop_count,
        "target_duration": target_duration,
    }
