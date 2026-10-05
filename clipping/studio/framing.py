"""Pure helpers for stable subject tracking and safe vertical crop geometry."""

import itertools
import math


def _finite(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def calculate_crop_size(source_width, source_height, target_ratio):
    """Return the largest source crop with the requested ratio, bounded to source."""
    width = int(source_width)
    height = int(source_height)
    ratio = _finite(target_ratio)
    if width <= 0 or height <= 0 or ratio is None or ratio <= 0:
        return 0, 0

    source_ratio = width / height
    if source_ratio >= ratio:
        crop_height = height
        if ratio < 0.75 and source_ratio >= 1.3:
            crop_height = int(height / 1.06)
        crop_width = int(crop_height * ratio)
    else:
        crop_width = width
        crop_height = int(width / ratio)

    crop_width = min(max(crop_width, 1), width)
    crop_height = min(max(crop_height, 1), height)
    return crop_width, crop_height


def clamp_crop_origin(center_x, center_y, crop_width, crop_height, frame_width, frame_height):
    """Clamp a crop origin so the requested window never leaves source bounds."""
    frame_width = max(int(frame_width), 1)
    frame_height = max(int(frame_height), 1)
    crop_width = min(max(int(crop_width), 1), frame_width)
    crop_height = min(max(int(crop_height), 1), frame_height)
    cx = _finite(center_x, frame_width / 2)
    cy = _finite(center_y, frame_height / 2)
    x = int(round(cx - crop_width / 2))
    y = int(round(cy - crop_height / 2))
    return (
        min(max(x, 0), frame_width - crop_width),
        min(max(y, 0), frame_height - crop_height),
    )


def _normalize_faces(faces, width, height, min_confidence):
    normalized = []
    for face in faces if isinstance(faces, list) else []:
        if not isinstance(face, dict):
            continue
        box = face.get("box")
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            continue
        values = [_finite(value) for value in box]
        score = _finite(face.get("score"), 1.0)
        if any(value is None for value in values) or score is None or score < min_confidence:
            continue
        x1, y1, x2, y2 = values
        x1 = min(max(x1, 0.0), float(width))
        x2 = min(max(x2, 0.0), float(width))
        y1 = min(max(y1, 0.0), float(height))
        y2 = min(max(y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        normalized.append({
            "box": (x1, y1, x2, y2),
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "area": area,
            "score": score,
            "quality": score * math.sqrt(area),
        })
    return normalized


def _make_target(faces, crop_width, group_fit_ratio):
    if not faces:
        return None

    single = max(faces, key=lambda face: face["quality"])
    group_targets = []
    max_count = min(len(faces), 8)
    for first, second in itertools.combinations(faces[:max_count], 2):
        x1 = min(first["box"][0], second["box"][0])
        y1 = min(first["box"][1], second["box"][1])
        x2 = max(first["box"][2], second["box"][2])
        y2 = max(first["box"][3], second["box"][3])
        if x2 - x1 > crop_width * group_fit_ratio:
            continue
        group_targets.append({
            "mode": "group",
            "box": (x1, y1, x2, y2),
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "members": ((first["cx"], first["cy"]), (second["cx"], second["cy"])),
            "quality": first["quality"] + second["quality"],
        })

    if group_targets:
        return max(group_targets, key=lambda target: target["quality"])
    return {
        "mode": "single",
        "box": single["box"],
        "cx": single["cx"],
        "cy": single["cy"],
        "members": ((single["cx"], single["cy"]),),
        "quality": single["quality"],
    }


def _target_distance(first, second):
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _match_target(current, faces, crop_width, association_distance):
    if current["mode"] == "group":
        candidates = []
        for first, second in itertools.combinations(faces[:8], 2):
            x1 = min(first["box"][0], second["box"][0])
            y1 = min(first["box"][1], second["box"][1])
            x2 = max(first["box"][2], second["box"][2])
            y2 = max(first["box"][3], second["box"][3])
            if x2 - x1 > crop_width * 0.85:
                continue
            members = ((first["cx"], first["cy"]), (second["cx"], second["cy"]))
            prior = current["members"]
            direct = _target_distance(prior[0], members[0]) + _target_distance(prior[1], members[1])
            crossed = _target_distance(prior[0], members[1]) + _target_distance(prior[1], members[0])
            distance = min(direct, crossed) / 2.0
            if distance <= association_distance:
                candidates.append((distance, {
                    "mode": "group",
                    "box": (x1, y1, x2, y2),
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                    "members": members,
                    "quality": first["quality"] + second["quality"],
                }))
        if candidates:
            return min(candidates, key=lambda item: (item[0], -item[1]["quality"]))[1]
        return None

    nearby = [
        face for face in faces
        if _target_distance((current["cx"], current["cy"]), (face["cx"], face["cy"])) <= association_distance
    ]
    if not nearby:
        return None
    selected = min(
        nearby,
        key=lambda face: (
            _target_distance((current["cx"], current["cy"]), (face["cx"], face["cy"])),
            -face["quality"],
        ),
    )
    return {
        "mode": "single",
        "box": selected["box"],
        "cx": selected["cx"],
        "cy": selected["cy"],
        "members": ((selected["cx"], selected["cy"]),),
        "quality": selected["quality"],
    }


def select_stable_face_track(
    samples,
    frame_width,
    frame_height,
    crop_width,
    min_confidence=0.55,
    sample_step=0.25,
    association_ratio=0.22,
    group_fit_ratio=0.85,
    target_hold_seconds=2.0,
    face_loss_hold_seconds=0.75,
    fallback_after_seconds=1.0,
):
    """Associate detections over time, holding targets through brief loss/outliers."""
    width = max(int(frame_width), 1)
    height = max(int(frame_height), 1)
    center_x, center_y = width / 2.0, height / 2.0
    min_confidence = min(max(_finite(min_confidence, 0.55), 0.0), 1.0)
    association_ratio = min(max(_finite(association_ratio, 0.22), 0.05), 0.4)
    sample_step = max(_finite(sample_step, 0.25), 0.01)
    target_hold_seconds = max(_finite(target_hold_seconds, 2.0), 0.0)
    face_loss_hold_seconds = max(_finite(face_loss_hold_seconds, 0.75), 0.0)
    fallback_after_seconds = max(
        _finite(fallback_after_seconds, 1.0),
        face_loss_hold_seconds,
    )
    group_fit_ratio = min(max(_finite(group_fit_ratio, 0.85), 0.1), 1.0)
    association_distance = width * association_ratio
    current = None
    selected_at = 0.0
    last_seen = None
    lost_since = None
    pending = None
    pending_count = 0
    track = []
    stats = {"max_faces": 0, "target_changes": 0, "held_samples": 0, "fallback_samples": 0}

    for index, sample in enumerate(samples if isinstance(samples, list) else []):
        if not isinstance(sample, dict):
            continue
        time = _finite(sample.get("time"), index * sample_step)
        faces = _normalize_faces(sample.get("faces"), width, height, min_confidence)
        stats["max_faces"] = max(stats["max_faces"], len(faces))
        target_changed = False
        holding = False
        fallback = False

        if current is None:
            current = _make_target(faces, crop_width, group_fit_ratio)
            if current is not None:
                selected_at = time
                last_seen = time
                cx, cy, box = current["cx"], current["cy"], current["box"]
            else:
                cx, cy, box = center_x, center_y, None
                fallback = True
        else:
            matched = _match_target(current, faces, crop_width, association_distance)
            if matched is not None:
                current = matched
                last_seen = time
                lost_since = None
                pending = None
                pending_count = 0
                cx, cy, box = current["cx"], current["cy"], current["box"]
            else:
                if lost_since is None:
                    lost_since = time
                proposed = _make_target(faces, crop_width, group_fit_ratio)
                if proposed is not None:
                    if pending is not None and pending["mode"] == proposed["mode"] and _target_distance(
                        (pending["cx"], pending["cy"]), (proposed["cx"], proposed["cy"])
                    ) <= association_distance:
                        pending_count += 1
                    else:
                        pending = proposed
                        pending_count = 1
                else:
                    pending = None
                    pending_count = 0

                lost_duration = time - last_seen if last_seen is not None else fallback_after_seconds
                confirmed = pending is not None and pending_count >= 2
                hold_elapsed = time - selected_at
                sustained_replacement = pending_count >= 4 and lost_duration >= face_loss_hold_seconds
                if (
                    confirmed
                    and lost_duration >= face_loss_hold_seconds
                    and (hold_elapsed >= target_hold_seconds or sustained_replacement)
                ):
                    current = pending
                    selected_at = time
                    last_seen = time
                    lost_since = None
                    pending = None
                    pending_count = 0
                    target_changed = True
                    stats["target_changes"] += 1
                    cx, cy, box = current["cx"], current["cy"], current["box"]
                elif proposed is None and lost_duration >= fallback_after_seconds:
                    cx, cy, box = center_x, center_y, None
                    fallback = True
                else:
                    cx, cy, box = current["cx"], current["cy"], current["box"]
                    holding = True

        if holding:
            stats["held_samples"] += 1
        if fallback:
            stats["fallback_samples"] += 1
        track.append({
            "time": time,
            "cx": cx,
            "cy": cy,
            "box": box,
            "holding": holding,
            "fallback": fallback,
            "target_changed": target_changed,
            "target_mode": current["mode"] if current is not None and not fallback else "center",
        })

    return track, stats


def _axis_step(current, target, deadzone, smooth_factor, max_step):
    delta = target - current
    if abs(delta) <= deadzone:
        return current
    desired = (delta - math.copysign(deadzone, delta)) * smooth_factor
    desired = min(max(desired, -max_step), max_step)
    return current + desired


def smooth_camera_track(
    track,
    frame_width,
    frame_height,
    crop_width,
    crop_height,
    deadzone_ratio=0.15,
    smooth_factor=0.30,
    max_step_ratio=0.08,
    headroom_ratio=0.20,
):
    """Smooth stabilized targets with dead-zone, speed limits, and headroom."""
    if not track:
        return []
    width = max(int(frame_width), 1)
    height = max(int(frame_height), 1)
    crop_width = min(max(int(crop_width), 1), width)
    crop_height = min(max(int(crop_height), 1), height)

    initial = [sample for sample in track[:5] if not sample.get("fallback")]
    initial = initial or list(track[:1])
    initial_x = sorted(sample["cx"] for sample in initial)
    initial_y = sorted(sample["cy"] for sample in initial)
    cam_x = initial_x[len(initial_x) // 2]
    cam_y = initial_y[len(initial_y) // 2]
    deadzone_x = crop_width * max(float(deadzone_ratio), 0.0)
    deadzone_y = crop_height * max(float(deadzone_ratio), 0.0)
    max_step_x = width * max(float(max_step_ratio), 0.0)
    max_step_y = height * max(float(max_step_ratio), 0.0)
    smooth_factor = min(max(float(smooth_factor), 0.0), 1.0)
    smooth = []

    for sample in track:
        target_x = _finite(sample.get("cx"), width / 2.0)
        target_y = _finite(sample.get("cy"), height / 2.0)
        box = sample.get("box")
        if box and crop_height < height:
            x1, y1, x2, y2 = box
            face_height = max(y2 - y1, 1.0)
            available = max(crop_height - face_height, 0.0)
            crop_top = y1 - available * headroom_ratio
            target_y = crop_top + crop_height / 2.0

        if sample.get("target_changed"):
            cam_x, cam_y = target_x, target_y
        else:
            cam_x = _axis_step(cam_x, target_x, deadzone_x, smooth_factor, max_step_x)
            cam_y = _axis_step(cam_y, target_y, deadzone_y, smooth_factor, max_step_y)

        cam_x = min(max(cam_x, crop_width / 2.0), width - crop_width / 2.0)
        cam_y = min(max(cam_y, crop_height / 2.0), height - crop_height / 2.0)
        smooth.append({**sample, "cx": cam_x, "cy": cam_y})
    return smooth
