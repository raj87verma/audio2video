"""Fetches free, real, deity/theme-specific photos & videos from Wikimedia
Commons -- a crowd-sourced media library (the same organization behind
Wikipedia) with genuine coverage of specific Hindu deities, temples and
ceremonies that generic stock-media libraries (Pexels/Pixabay) simply
don't have.

Why this exists: Pexels/Pixabay have essentially zero coverage of named
deities -- searching "Khatu Shyam" on either returns nothing, so even
perfect keyword extraction can only ever fetch generic "temple"/"prayer"
stock photography from those sources, never the actual deity a devotional
song is about. Wikimedia Commons is different: it's populated by
devotees photographing/filming their own temples, murtis and ceremonies,
so it has real, on-theme, specific coverage (verified: "Khatu Shyam"
alone returns 49 real hits including the deity's own murti; "aarti
filetype:video" returns real aarti ceremony footage).

This module is used *ahead of* Pexels/Pixabay specifically for detected
deity queries (see content_hints.py + pipeline.py) -- generic devotional
keywords ("hindu temple", "diya lamp", ...) still go to Pexels/Pixabay as
before, since those work fine for generic imagery and Wikimedia isn't
meant to replace them wholesale.

Compliance notes (see https://w.wiki/4wJS, the Wikimedia robot policy):
  - Every request identifies itself with a real, descriptive User-Agent
    including a contact URL, as required -- a vague/generic User-Agent
    gets a 403 ("Please honor our robot policy"), verified directly
    against the live API while building this module.
  - The Action API (search) is called with concurrency 1 and only when
    needed (results are cached per query for the run -- see
    pipeline.py's media resolver, which this module is designed to slot
    into the same way as media_fetcher.py).
  - Media downloads (upload.wikimedia.org) are throttled to a small
    concurrency and don't need to be parallelized for our per-shot usage
    pattern (one download at a time is already how media_fetcher.py's
    download_asset() is used).
  - No content is cached/redistributed beyond the same on-disk media
    cache directory Pexels/Pixabay assets already use, purely for reuse
    within a single render.

License note: unlike Pexels/Pixabay (which require no attribution),
Wikimedia Commons media is almost always under a Creative Commons license
that DOES require attribution (typically CC BY-SA -- verified via the
`extmetadata` API field on real results). This module always returns the
attribution fields (`author`, `license_name`, `license_url`,
`source_url`) alongside the asset, and callers (see pipeline.py /
credits.py) are responsible for surfacing them to the end user.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from ..config import MEDIA_CACHE_DIR

log = logging.getLogger(__name__)

API_URL = "https://commons.wikimedia.org/w/api.php"
REQUEST_TIMEOUT = 20
DOWNLOAD_TIMEOUT = 60

# Wikimedia's robot policy requires a descriptive User-Agent identifying
# the application and a way to contact its operator; a vague one (e.g.
# just an app name + fake email) gets a 403 ("Please honor our robot
# policy") -- verified directly against the live API. This exact format
# (app name/version, a real project URL, and a contact method) is what
# was confirmed to work.
USER_AGENT = "Audio2Video/0.1 (https://github.com/raj87verma/audio2video; contact via GitHub issues)"

# Only fetch images/videos Commons itself classifies under these MIME
# top-types -- excludes audio, PDFs, and other non-visual media types
# that can otherwise show up in a broad text search.
_VISUAL_MIME_PREFIXES = ("image/", "video/")

# File extensions we can actually feed into MoviePy/FFmpeg without any
# extra handling -- Commons also hosts .tif/.svg/.pdf-adjacent formats
# for images and .ogv alongside .webm for video, all of which FFmpeg can
# decode, but we deliberately keep this list to the common, reliably-
# decodable formats actually seen in practice to avoid edge cases.
_SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_SUPPORTED_VIDEO_EXTS = {".webm", ".ogv"}


@dataclass
class WikimediaAsset:
    kind: str            # 'video' or 'image'
    url: str              # direct upload.wikimedia.org download URL
    title: str            # Commons "File:..." title, human-readable
    width: int
    height: int
    duration: float | None  # seconds, only for videos
    query: str
    # Attribution -- Commons media is almost always CC-licensed with an
    # attribution requirement, unlike Pexels/Pixabay. Always populated
    # (falling back to sensible placeholders) so callers can build a
    # credits list without special-casing missing fields.
    author: str = "Unknown"
    license_name: str = "Unknown license"
    license_url: str = ""
    source_url: str = ""
    local_path: str | None = None  # populated after download_asset()


def _cache_key(query: str, kind: str) -> str:
    digest = hashlib.sha1(f"wikimedia:{kind}:{query}".encode("utf-8")).hexdigest()[:16]
    safe_query = "".join(c if c.isalnum() else "_" for c in query)[:30]
    return f"wikimedia_{kind}_{safe_query}_{digest}"


def _extension_for(title: str) -> str:
    return Path(title).suffix.lower()


def search_wikimedia(query: str, kind: str = "image", limit: int = 8) -> list[WikimediaAsset]:
    """Search Wikimedia Commons for real, on-theme media matching `query`.

    `kind` is 'image' or 'video' -- Commons doesn't have separate search
    endpoints for these like Pexels/Pixabay do, so this appends a
    `filetype:` search operator (verified against the live API: e.g.
    "aarti filetype:video" correctly returns only video files).

    Returns [] on any error, on no results, or for entries whose file
    extension isn't one we know how to hand to MoviePy/FFmpeg -- callers
    should treat that exactly like "nothing found" and fall back to
    another source (Pexels/Pixabay, then procedural visuals), never raise.
    """
    filetype = "video" if kind == "video" else "bitmap"
    search_query = f"{query} filetype:{filetype}"
    try:
        resp = requests.get(
            API_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": search_query,
                "srnamespace": 6,  # "File:" namespace
                "srlimit": limit,
                "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        titles = [hit["title"] for hit in data.get("query", {}).get("search", [])]
        if not titles:
            return []
        return _fetch_imageinfo(titles, kind, query)
    except Exception as exc:
        log.warning("Wikimedia %s search failed for %r: %s", kind, query, exc)
        return []


def _fetch_imageinfo(titles: list[str], kind: str, query: str) -> list[WikimediaAsset]:
    """Resolve a batch of Commons "File:..." titles to their direct
    download URLs + license metadata via a single `imageinfo` call
    (batched, not one request per title, to stay well within the
    Action API's request-rate guidance).
    """
    supported_exts = _SUPPORTED_VIDEO_EXTS if kind == "video" else _SUPPORTED_IMAGE_EXTS
    try:
        resp = requests.get(
            API_URL,
            params={
                "action": "query",
                "titles": "|".join(titles),
                "prop": "imageinfo",
                "iiprop": "url|size|mime|extmetadata",
                "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.warning("Wikimedia imageinfo lookup failed for %r: %s", titles, exc)
        return []

    assets: list[WikimediaAsset] = []
    for page in data.get("query", {}).get("pages", {}).values():
        title = page.get("title", "")
        if _extension_for(title) not in supported_exts:
            continue
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        mime = info.get("mime", "")
        if not mime.startswith(_VISUAL_MIME_PREFIXES):
            continue

        meta = info.get("extmetadata", {}) or {}

        def _meta(field: str, default: str = "") -> str:
            value = meta.get(field, {})
            raw = value.get("value", default) if isinstance(value, dict) else default
            # Some fields (notably Artist/Credit) contain raw HTML links;
            # strip tags for a plain-text attribution string rather than
            # leaking markup into the UI/credits file.
            return _strip_html(str(raw)) if raw else default

        assets.append(WikimediaAsset(
            kind=kind,
            url=info.get("url", ""),
            title=title,
            width=info.get("width", 0),
            height=info.get("height", 0),
            duration=info.get("duration"),
            query=query,
            author=_meta("Artist", "Unknown") or "Unknown",
            license_name=_meta("LicenseShortName", "Unknown license") or "Unknown license",
            license_url=_meta("LicenseUrl", ""),
            source_url=f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
        ))
    return assets


def _strip_html(text: str) -> str:
    """Minimal HTML-tag stripper for Commons' Artist/Credit metadata
    fields, which frequently contain an `<a href=...>Name</a>` link
    rather than plain text. Deliberately simple (regex, not a full HTML
    parser) since this only ever needs to handle short attribution
    strings, not arbitrary markup.
    """
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def download_asset(asset: WikimediaAsset, cache_dir: Path = MEDIA_CACHE_DIR) -> str | None:
    """Download `asset` to the local media cache, returning the local path.

    Mirrors `media_fetcher.download_asset`'s caching behavior (content-
    hash-based filename, skip re-download if already cached) so repeated
    runs/shots reuse files instead of re-fetching them.
    """
    if not asset.url:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.sha1(asset.url.encode("utf-8")).hexdigest()[:12]
    ext = _extension_for(asset.title) or (".mp4" if asset.kind == "video" else ".jpg")
    filename = f"{_cache_key(asset.query, asset.kind)}_{url_hash}{ext}"
    local_path = cache_dir / filename

    if local_path.exists() and local_path.stat().st_size > 0:
        asset.local_path = str(local_path)
        return asset.local_path

    try:
        with requests.get(asset.url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=DOWNLOAD_TIMEOUT) as resp:
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        asset.local_path = str(local_path)
        log.info("Downloaded Wikimedia %s asset for %r -> %s", asset.kind, asset.query, local_path.name)
        return asset.local_path
    except Exception as exc:
        log.warning("Failed to download Wikimedia asset for %r (%s): %s", asset.query, asset.url, exc)
        if local_path.exists():
            local_path.unlink(missing_ok=True)
        return None


def fetch_best_asset(query: str, kind: str = "image") -> WikimediaAsset | None:
    """Convenience: search + download the first working Wikimedia asset
    for `query`. Tries candidates in order until one downloads
    successfully, so a single broken/oversized file doesn't sink the
    whole shot. Returns None (never raises) if nothing usable is found.
    """
    candidates = search_wikimedia(query, kind=kind)
    for asset in candidates:
        if download_asset(asset):
            return asset
    return None
