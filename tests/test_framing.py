import importlib.util
from pathlib import Path


_framing_path = Path(__file__).resolve().parents[1] / "clipping" / "studio" / "framing.py"
_framing_spec = importlib.util.spec_from_file_location("test_framing_helpers", _framing_path)
_framing = importlib.util.module_from_spec(_framing_spec)
_framing_spec.loader.exec_module(_framing)
calculate_crop_size = _framing.calculate_crop_size
clamp_crop_origin = _framing.clamp_crop_origin
select_stable_face_track = _framing.select_stable_face_track
smooth_camera_track = _framing.smooth_camera_track


def _face(center_x, center_y=500, box_width=80, box_height=120, score=0.9):
    return {
        "box": (
            center_x - box_width / 2,
            center_y - box_height / 2,
            center_x + box_width / 2,
            center_y + box_height / 2,
        ),
        "score": score,
    }


def _sample(time, *faces):
    return {"time": time, "faces": list(faces)}


def test_temporal_association_keeps_target_when_other_face_is_larger():
    track, stats = select_stable_face_track(
        [
            _sample(0.0, _face(300, score=0.7)),
            _sample(0.25, _face(310, score=0.6), _face(1500, score=0.99, box_width=200, box_height=250)),
            _sample(0.5, _face(320, score=0.6), _face(1480, score=0.99, box_width=220, box_height=270)),
        ],
        frame_width=1920,
        frame_height=1080,
        crop_width=607,
    )

    assert [round(item["cx"]) for item in track] == [300, 310, 320]
    assert stats["target_changes"] == 0


def test_one_frame_outlier_is_held_and_then_ignored():
    track, stats = select_stable_face_track(
        [
            _sample(0.0, _face(300)),
            _sample(0.25, _face(1500)),
            _sample(0.5, _face(310)),
        ],
        frame_width=1920,
        frame_height=1080,
        crop_width=607,
    )

    assert track[1]["holding"] is True
    assert track[1]["cx"] == 300
    assert track[2]["cx"] == 310
    assert stats["target_changes"] == 0


def test_brief_face_loss_holds_and_long_loss_falls_back():
    track, stats = select_stable_face_track(
        [_sample(0.0, _face(300))]
        + [_sample(time) for time in (0.25, 0.5, 0.75, 1.0, 1.25)],
        frame_width=1920,
        frame_height=1080,
        crop_width=607,
    )

    assert track[2]["holding"] is True
    assert track[-1]["fallback"] is True
    assert track[-1]["cx"] == 960
    assert stats["fallback_samples"] > 0


def test_persistent_new_subject_switches_only_after_confirmation_and_hold():
    samples = [_sample(0.0, _face(300))]
    samples.extend(_sample(time, _face(1500)) for time in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0))

    track, stats = select_stable_face_track(
        samples,
        frame_width=1920,
        frame_height=1080,
        crop_width=607,
    )

    assert track[1]["cx"] == 300
    changed_at = next(index for index, item in enumerate(track) if item["target_changed"])
    assert track[changed_at]["time"] <= 1.0
    assert track[-1]["cx"] == 1500
    assert stats["target_changes"] == 1


def test_close_faces_are_grouped_when_the_crop_can_include_both():
    track, _ = select_stable_face_track(
        [_sample(0.0, _face(700), _face(1000))],
        frame_width=1920,
        frame_height=1080,
        crop_width=607,
    )

    assert track[0]["target_mode"] == "group"
    assert track[0]["cx"] == 850


def test_low_confidence_face_uses_center_fallback():
    track, _ = select_stable_face_track(
        [_sample(0.0, _face(300, score=0.3))],
        frame_width=1920,
        frame_height=1080,
        crop_width=607,
        min_confidence=0.55,
    )

    assert track[0]["fallback"] is True
    assert track[0]["cx"] == 960


def test_crop_uses_face_box_for_headroom_when_vertical_pan_is_available():
    track = [{
        "time": 0.0,
        "cx": 450,
        "cy": 400,
        "box": (400, 300, 500, 500),
        "target_changed": False,
    }]
    smooth = smooth_camera_track(
        track,
        frame_width=1000,
        frame_height=2000,
        crop_width=560,
        crop_height=1600,
    )
    _, crop_y = clamp_crop_origin(
        smooth[0]["cx"], smooth[0]["cy"], 560, 1600, 1000, 2000
    )

    assert crop_y == 0
    assert track[0]["box"][1] - crop_y >= 0


def test_crop_geometry_handles_wide_portrait_square_and_edge_bounds():
    assert calculate_crop_size(1920, 1080, 9 / 16) == (572, 1018)
    assert calculate_crop_size(1080, 2400, 9 / 16) == (1080, 1920)
    assert calculate_crop_size(1080, 1080, 9 / 16) == (607, 1080)
    assert calculate_crop_size(1080, 1920, 9 / 16) == (1080, 1920)
    assert clamp_crop_origin(0, 0, 607, 1080, 1920, 1080) == (0, 0)


def test_camera_deadzone_and_speed_limit_are_applied():
    samples = [
        {"time": 0.0, "cx": 500, "cy": 500, "box": None, "target_changed": False},
        {"time": 0.25, "cx": 520, "cy": 500, "box": None, "target_changed": False},
        {"time": 0.5, "cx": 900, "cy": 500, "box": None, "target_changed": False},
    ]

    smooth = smooth_camera_track(
        samples,
        frame_width=1000,
        frame_height=800,
        crop_width=300,
        crop_height=800,
        deadzone_ratio=0.15,
        smooth_factor=0.3,
        max_step_ratio=0.08,
    )

    assert smooth[1]["cx"] == smooth[0]["cx"]
    assert smooth[2]["cx"] - smooth[1]["cx"] <= 80
