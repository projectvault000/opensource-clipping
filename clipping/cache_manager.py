from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_CACHE_ROOT = "outputs/cache"


def _stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class CacheManager:
    """File-backed cache for expensive, deterministic pipeline stages."""

    def __init__(self, base_dir: str | os.PathLike[str] = DEFAULT_CACHE_ROOT):
        raw_base = str(base_dir)
        cache_root = Path(raw_base)
        if cache_root.name != "cache":
            cache_root = cache_root / "cache"
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._stats: dict[str, dict[str, int]] = {}
        self._stage_timings: dict[str, float] = {}

    @staticmethod
    def make_key(category: str, **payload: Any) -> str:
        clean = {str(key): value for key, value in sorted(payload.items()) if value is not None}
        digest = hashlib.sha256(_stable_json({"category": category, "payload": clean}).encode("utf-8")).hexdigest()
        return digest[:32]

    def category_dir(self, category: str) -> Path:
        path = self.cache_root / category
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_path(self, category: str, key: str) -> Path:
        return self.category_dir(category) / f"{key}.json"

    def get_json(self, category: str, key: str) -> Any | None:
        stats = self._stats.setdefault(category, {"hits": 0, "misses": 0})
        path = self.artifact_path(category, key)
        if not path.exists():
            stats["misses"] += 1
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            stats["misses"] += 1
            return None
        stats["hits"] += 1
        return payload

    def put_json(self, category: str, key: str, payload: Any) -> str:
        path = self.artifact_path(category, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix="cache_", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return str(path)

    def record_timing(self, stage_name: str, duration_seconds: float) -> float:
        value = float(duration_seconds or 0.0)
        self._stage_timings[stage_name] = value
        return value

    def get_stats(self) -> dict[str, Any]:
        stats = {category: {"hits": values.get("hits", 0), "misses": values.get("misses", 0)} for category, values in self._stats.items()}
        stats["stage_timings"] = dict(self._stage_timings)
        return stats


__all__ = ["CacheManager"]
