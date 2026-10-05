import html
import importlib.util
import json
import math
import os
import random
import re
import shutil
import string
import subprocess
import textwrap
import time
import urllib.parse
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont
from yt_dlp import YoutubeDL

FIREFOX_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0"

def _load_studio_internal_module(file_name: str, module_alias: str):
    module_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_helpers = _load_studio_internal_module("helpers.py", "clipping_studio_helpers")
_ffmpeg_utils = _load_studio_internal_module("ffmpeg_utils.py", "clipping_studio_ffmpeg_utils")
format_seconds = _helpers.format_seconds
escape_ffmpeg_filter_value = _helpers.escape_ffmpeg_filter_value
detect_video_encoder = _ffmpeg_utils.detect_video_encoder
get_ts_encode_args = _ffmpeg_utils.get_ts_encode_args
get_mp4_encode_args = _ffmpeg_utils.get_mp4_encode_args
open_ffmpeg_video_writer = _ffmpeg_utils.open_ffmpeg_video_writer
build_ffmpeg_progress_cmd = _ffmpeg_utils.build_ffmpeg_progress_cmd
run_ffmpeg_with_progress = _ffmpeg_utils.run_ffmpeg_with_progress


def buat_thumbnail(video_path, output_image_path, teks, cfg, candidate_times=None):
    """
    Extract a frame from the video, composite the clip title on top, and save as a thumbnail image.

    Args:
        input_video (str): Path to the source video file.
        output_image (str): Destination path for the generated JPEG thumbnail.
        start_clip (float): Clip start time in seconds (to locate a frame).
        end_clip (float): Clip end time in seconds.
        judul (str): The title text to be written on the thumbnail.
        rasio (str): Target output ratio string ('9:16' or '16:9').
        cfg: Runtime configuration object.

    Returns:
        str: The path to the created image, or None if creation fails.

    Side Effects:
        Uses `cv2.VideoCapture` to extract a frame from `input_video`.
        Writes a JPEG file to `output_image`.

    Raises:
        Exception: If image processing or saving fails.
    """
    if not os.path.exists(cfg.file_font_thumbnail):
        urllib.request.urlretrieve(cfg.url_font_thumbnail, cfg.file_font_thumbnail)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = frame_count / fps if fps > 0 else 0.0
    times = candidate_times or [0.0, duration * 0.15, duration * 0.35, duration * 0.55]
    best_frame = None
    best_time = None
    best_score = float("-inf")
    for candidate_time in times:
        safe_time = min(max(float(candidate_time), 0.0), max(duration - 0.05, 0.0))
        cap.set(cv2.CAP_PROP_POS_MSEC, safe_time * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(gray.mean())
        if mean_brightness < 12.0:
            continue
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        score = min(sharpness, 5000.0) + min(mean_brightness, 180.0) * 0.25
        if score > best_score:
            best_score = score
            best_frame = frame.copy()
            best_time = safe_time
    cap.release()

    if best_frame is None:
        return None

    frame = best_frame

    img = Image.alpha_composite(
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA"),
        Image.new("RGBA", (frame.shape[1], frame.shape[0]), (0, 0, 0, 128)),
    ).convert("RGB")

    draw = ImageDraw.Draw(img)
    font_sz = int(img.size[0] * 0.12)
    font = ImageFont.truetype(cfg.file_font_thumbnail, font_sz)
    lines = textwrap.wrap(teks, width=12)

    y_text = (img.size[1] - (len(lines) * (font_sz + 10))) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        x_text = (img.size[0] - line_w) // 2
        draw.text(
            (x_text, y_text),
            line,
            font=font,
            fill="white",
            stroke_width=5,
            stroke_fill="black",
        )
        y_text += font_sz + 10

    img.save(output_image_path)
    return best_time


