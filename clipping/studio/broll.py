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
import urllib.error

import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont
from yt_dlp import YoutubeDL

from clipping.broll_policy import (
    cache_asset_path,
    get_used_asset_ids,
    mark_asset_used,
    normalize_broll_query,
    normalize_visual_intent,
    probe_broll_media,
    rank_pexels_candidates,
    record_asset_metadata,
    simplify_broll_query,
)

def _load_studio_internal_module(file_name: str, module_alias: str):
    module_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

utils = _load_studio_internal_module("utils.py", "clipping_studio_utils")
_resize_frame = utils._resize_frame
_is_vertical_ratio = utils._is_vertical_ratio

FIREFOX_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0"


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

USED_PEXELS_IDS = set()  # Retained for compatibility; job state lives in broll_policy.


def reset_broll_session(output_dir):
    """Reset per-job Pexels asset reuse and manifest metadata."""
    from clipping.broll_policy import reset_broll_session as _reset

    _reset(output_dir)


def get_broll_asset_metadata(output_dir, output_path):
    """Return non-secret provider/asset metadata for a successfully downloaded file."""
    from clipping.broll_policy import get_asset_metadata

    return get_asset_metadata(output_dir, output_path)


def download_pexels_broll(
    query,
    rasio,
    output_filename,
    pexels_api_key,
    minimum_duration=0.0,
    output_dir=None,
    visual_intent=None,
):
    """
    Search and download one Pexels B-roll video clip matching the query and aspect ratio.

    Args:
        query (str): Search query term (e.g., 'nature', 'technology').
        rasio (str): Target aspect ratio string (`9:16` for portrait or `16:9` for landscape).
        output_filename (str): Local file path where the downloaded MP4 will be saved.
        pexels_api_key (str): Valid Pexels API key for authorization.

    Returns:
        bool: True if the video was successfully downloaded and saved, False otherwise.

    Side Effects:
        Makes HTTP GET requests to the Pexels API and video CDN.
        Mutates the global `USED_PEXELS_IDS` set to prevent duplicate downloads.
        Writes a temporary file (`.part`) and renames it upon successful download.
        Prints status and error messages to stdout.

    Raises:
        None explicitly. Exceptions during download or API calls are caught and return False.
    """
    if not pexels_api_key:
        print("   ⚠️ PEXELS_API_KEY tidak ditemukan. B-roll dilewati.")
        return False

    safe_query = normalize_broll_query(query)
    if safe_query is None:
        print("   ⚠️ B-roll query ditolak karena kosong, terlalu umum, atau tidak aman.")
        return False

    orientation = "portrait" if _is_vertical_ratio(rasio) else "landscape"
    session_dir = output_dir or os.path.dirname(os.path.abspath(output_filename)) or os.getcwd()
    minimum_duration = max(float(minimum_duration or 0.0), 0.0)
    search_queries = [safe_query]
    simplified = simplify_broll_query(safe_query)
    if simplified:
        search_queries.append(simplified)

    for query_index, search_query in enumerate(search_queries):
        params = urllib.parse.urlencode({
            "query": search_query,
            "orientation": orientation,
            "per_page": 30,
            "size": "large",
            "resolution_name": "1080p",
        })
        req = urllib.request.Request(
            f"https://api.pexels.com/videos/search?{params}",
            headers={"Authorization": pexels_api_key, "User-Agent": "Mozilla/5.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            print(f"   ⚠️ Pexels search failed (HTTP {exc.code}); source footage will be used.")
            return False
        except Exception as exc:
            print(f"   ⚠️ Pexels search failed ({type(exc).__name__}); source footage will be used.")
            return False

        videos = data.get("videos", []) if isinstance(data, dict) else []
        candidates = rank_pexels_candidates(
            videos,
            target_ratio=(9 / 16 if orientation == "portrait" else 16 / 9),
            minimum_duration=minimum_duration,
            used_ids=get_used_asset_ids(session_dir),
            max_candidates=3,
        )
        if not candidates:
            if query_index == 0 and len(search_queries) > 1:
                print("   ⚠️ No usable B-roll result; trying one concise query variant.")
                continue
            print(f"   ⚠️ Pexels returned no usable video for '{search_query}'.")
            return False

        for candidate in candidates:
            cache_path = cache_asset_path(session_dir, candidate["asset_id"], candidate["file_url"])
            cached_media = probe_broll_media(cache_path, minimum_duration)
            if cached_media:
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
                    shutil.copy2(cache_path, output_filename)
                    mark_asset_used(session_dir, candidate["asset_id"])
                    record_asset_metadata(session_dir, output_filename, {
                        "provider": "pexels",
                        "asset_id": candidate["asset_id"],
                        "asset_url": candidate["video_url"],
                        "file_url": candidate["file_url"],
                        "query": search_query,
                        "visual_intent": normalize_visual_intent(visual_intent),
                        **cached_media,
                        "cache_hit": True,
                    })
                    return True
                except OSError:
                    pass

            temp_path = output_filename + ".part"
            try:
                download_req = urllib.request.Request(
                    candidate["file_url"], headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(download_req, timeout=20) as response, open(temp_path, "wb") as out_file:
                    shutil.copyfileobj(response, out_file)
                media = probe_broll_media(temp_path, minimum_duration)
                if not media:
                    print(f"   ⚠️ Pexels asset {candidate['asset_id']} failed media validation; trying another result.")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    continue

                os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
                os.replace(temp_path, output_filename)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                cache_temp = cache_path + ".part"
                shutil.copy2(output_filename, cache_temp)
                os.replace(cache_temp, cache_path)
                mark_asset_used(session_dir, candidate["asset_id"])
                record_asset_metadata(session_dir, output_filename, {
                    "provider": "pexels",
                    "asset_id": candidate["asset_id"],
                    "asset_url": candidate["video_url"],
                    "file_url": candidate["file_url"],
                    "query": search_query,
                    "visual_intent": normalize_visual_intent(visual_intent),
                    **media,
                    "cache_hit": False,
                })
                return True
            except Exception as exc:
                print(f"   ⚠️ Pexels asset download failed ({type(exc).__name__}); trying another result.")
                for partial in (temp_path, temp_path + ".part"):
                    if os.path.exists(partial):
                        os.remove(partial)

        if query_index == 0 and len(search_queries) > 1:
            print("   ⚠️ B-roll candidates failed download/validation; trying one concise query variant.")

    print("   ⚠️ No valid B-roll asset survived the bounded attempts.")
    return False


def crop_center_broll(img, target_w, target_h):
    """
    Center-crop an image frame to the exact target aspect ratio, then resize it.

    Args:
        img (np.ndarray): Input image frame array (from OpenCV).
        target_w (int): Desired output width in pixels.
        target_h (int): Desired output height in pixels.

    Returns:
        np.ndarray: The cropped and resized frame.

    Side Effects:
        None.

    Raises:
        cv2.error: If the input image format is invalid or resizing fails.
    """
    h, w = img.shape[:2]
    target_ratio = target_w / target_h
    img_ratio = w / h

    if img_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x = (w - new_w) // 2
        img = img[:, x : x + new_w]
    elif img_ratio < target_ratio:
        new_h = int(w / target_ratio)
        y = (h - new_h) // 2
        img = img[y : y + new_h, :]

    return _resize_frame(img, (target_w, target_h))


