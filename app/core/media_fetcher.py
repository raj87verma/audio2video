"""Fetches free, royalty-free stock photos/videos for a shot's keywords.

Two free providers are supported, both requiring only a free sign-up
(no credit card) to obtain an API key:

  - Pexels    https://www.pexels.com/api/       (photos + videos)
  - Pixabay   https://pixabay.com/api/docs/      (photos + videos)

Both are queried per-keyword; results are cached to disk (by query string)
so re-runs and repeated keywords across shots don't re-download the same
asset or exceed rate limits. If neither API key is configured, or a query
returns nothing, this module returns an empty list and the caller
(pipeline) falls back to the procedural visual generator instead of
failing.

License note: both Pexels and Pixabay content is free to use, including
commercially, without attribution required by their license terms —
still, always follow the up-to-date license text on their sites for any
redistribution.
"""
from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import requests

from ..config import MEDIA_CACHE_DIR, Settings

log = logging.getLogger(__name__)

REQUEST_TIMEOUT = 20
USER_AGENT = "Audio2Video/0.1 (+https://github.com/) free-stock-media-client"


@dataclass
class MediaAsset:
    kind: str          # 'video' or 'image'
    url: str            # direct download URL
    width: int
    height: int
    duration: float | None  # seconds, only for videos
    source: str          # 'pexels' or 'pixabay'
    query: str
    local_path: str | None = None  # populated after download()


def _cache_key(query: str, kind: str, source: str) -> str:
    digest = hashlib.sha1(f"{source}:{kind}:{query}".encode("utf-8")).hexdigest()[:16]
    safe_query = "".join(c if c.isalnum() else "_" for c in query)[:30]
    return f"{source}_{kind}_{safe_query}_{digest}"


def _pexels_search(query: str, api_key: str, kind: str, per_page: int = 8) -> list[MediaAsset]:
    headers = {"Authorization": api_key, "User-Agent": USER_AGENT}
    try:
        if kind == "video":
            resp = requests.get(
                "https://api.pexels.com/videos/search",
                params={"query": query, "per_page": per_page, "orientation": "landscape"},
                headers=headers, timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            assets = []
            for video in data.get("videos", []):
                files = sorted(
                    (vf for vf in video.get("video_files", []) if vf.get("width")),
                    key=lambda vf: vf.get("width", 0),
                )
                # pick a moderate-resolution file (avoid huge 4k downloads)
                candidates = [vf for vf in files if 960 <= vf.get("width", 0) <= 1920] or files
                if not candidates:
                    continue
                vf = candidates[-1]
                assets.append(MediaAsset(
                    kind="video", url=vf["link"], width=vf.get("width", 0), height=vf.get("height", 0),
                    duration=video.get("duration"), source="pexels", query=query,
                ))
            return assets
        else:
            resp = requests.get(
                "https://api.pexels.com/v1/search",
                params={"query": query, "per_page": per_page, "orientation": "landscape"},
                headers=headers, timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            assets = []
            for photo in data.get("photos", []):
                src = photo.get("src", {})
                url = src.get("large2x") or src.get("large") or src.get("original")
                if not url:
                    continue
                assets.append(MediaAsset(
                    kind="image", url=url, width=photo.get("width", 0), height=photo.get("height", 0),
                    duration=None, source="pexels", query=query,
                ))
            return assets
    except Exception as exc:
        log.warning("Pexels %s search failed for %r: %s", kind, query, exc)
        return []


def _pixabay_search(query: str, api_key: str, kind: str, per_page: int = 8) -> list[MediaAsset]:
    try:
        resp = requests.get(
            "https://pixabay.com/api/videos/" if kind == "video" else "https://pixabay.com/api/",
            params={
                "key": api_key,
                "q": query,
                "per_page": per_page,
                "safesearch": "true",
                **({"orientation": "horizontal"} if kind != "video" else {}),
            },
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        assets = []
        if kind == "video":
            for hit in data.get("hits", []):
                videos = hit.get("videos", {})
                variant = videos.get("medium") or videos.get("small") or videos.get("large")
                if not variant or not variant.get("url"):
                    continue
                assets.append(MediaAsset(
                    kind="video", url=variant["url"], width=variant.get("width", 0),
                    height=variant.get("height", 0), duration=hit.get("duration"),
                    source="pixabay", query=query,
                ))
        else:
            for hit in data.get("hits", []):
                url = hit.get("largeImageURL") or hit.get("webformatURL")
                if not url:
                    continue
                assets.append(MediaAsset(
                    kind="image", url=url, width=hit.get("imageWidth", 0), height=hit.get("imageHeight", 0),
                    duration=None, source="pixabay", query=query,
                ))
        return assets
    except Exception as exc:
        log.warning("Pixabay %s search failed for %r: %s", kind, query, exc)
        return []


def search_media(query: str, settings: Settings, prefer_video: bool = True) -> list[MediaAsset]:
    """Search all configured providers for `query`, video-first if preferred.

    Returns a combined, shuffled-within-provider list of MediaAsset (not yet
    downloaded). Empty list if no providers configured or nothing found.
    """
    results: list[MediaAsset] = []
    kinds = ["video", "image"] if prefer_video else ["image", "video"]

    for kind in kinds:
        kind_results: list[MediaAsset] = []
        if settings.pexels_api_key:
            kind_results.extend(_pexels_search(query, settings.pexels_api_key, kind))
        if settings.pixabay_api_key:
            kind_results.extend(_pixabay_search(query, settings.pixabay_api_key, kind))
        if kind_results:
            random.shuffle(kind_results)
            results.extend(kind_results)

    return results


def download_asset(asset: MediaAsset, cache_dir: Path = MEDIA_CACHE_DIR) -> str | None:
    """Download `asset` to the local media cache, returning the local path.

    Uses a content-addressed-ish cache key (provider+kind+query hash) plus
    the asset's own URL hash, so repeated runs reuse previously downloaded
    files instead of re-fetching them.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.sha1(asset.url.encode("utf-8")).hexdigest()[:12]
    ext = "mp4" if asset.kind == "video" else "jpg"
    filename = f"{_cache_key(asset.query, asset.kind, asset.source)}_{url_hash}.{ext}"
    local_path = cache_dir / filename

    if local_path.exists() and local_path.stat().st_size > 0:
        asset.local_path = str(local_path)
        return asset.local_path

    try:
        with requests.get(asset.url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        asset.local_path = str(local_path)
        log.info("Downloaded %s asset for %r -> %s", asset.kind, asset.query, local_path.name)
        return asset.local_path
    except Exception as exc:
        log.warning("Failed to download asset for %r (%s): %s", asset.query, asset.url, exc)
        if local_path.exists():
            local_path.unlink(missing_ok=True)
        return None


def fetch_best_asset(query: str, settings: Settings, prefer_video: bool = True) -> MediaAsset | None:
    """Convenience: search + download the first working asset for `query`.

    Tries candidates in order until one downloads successfully, so a single
    broken/expired URL doesn't sink the whole shot.
    """
    candidates = search_media(query, settings, prefer_video=prefer_video)
    for asset in candidates:
        if download_asset(asset):
            return asset
    return None
