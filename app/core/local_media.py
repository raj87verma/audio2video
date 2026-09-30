"""Lets a user point Audio2Video at a single local folder of their own
royalty-free photos/videos and use ONLY that folder for every song's
visuals -- permanently disabling Wikimedia Commons, Pexels, and Pixabay
entirely, for every song, not just devotional/deity ones.

Why this exists (and how it differs from `user_media.py`): `user_media.py`
is deity-aware -- it only kicks in for a song whose filename/lyrics name a
specific recognized deity, and only bypasses online sources for THAT song
if its matching per-deity subfolder happens to have files in it; every
other song (or an empty subfolder) still falls back to Wikimedia/Pexels/
Pixabay as usual. Real user feedback was that this wasn't what they
wanted at all: they explicitly do not want the app calling out to
Pexels/Pixabay/Wikimedia under any circumstance, for any song, once
they've supplied their own asset folder -- a single, simple, permanent
switch, not a per-song/per-deity fallback chain.

This module implements exactly that: a single folder path (stored in
`Settings.local_media_dir`), with two fixed subfolders:

    <local_media_dir>/
      images/    <- any number of photos
      videos/    <- any number of video clips

`ensure_local_media_subdirs()` creates these two subfolders under
whatever path the user has chosen (idempotent, safe to call repeatedly).
`scan_local_media_dir()` returns every usable image/video found in them.

Callers (see `pipeline.py`) are responsible for treating
`Settings.local_media_dir` being non-empty as an unconditional, permanent
switch: once set, Wikimedia/Pexels/Pixabay/deity-specific `user_media.py`
must never be consulted for any song, even if this folder is currently
empty (in which case shots simply fall back to the procedural generator,
exactly like "no stock media available" already does elsewhere in this
app) -- there is deliberately no "folder empty -> fall back online" path
here, unlike `user_media.py`'s per-deity behavior.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .user_media import _SUPPORTED_IMAGE_EXTS, _SUPPORTED_VIDEO_EXTS, UserMediaAsset

log = logging.getLogger(__name__)

IMAGES_SUBDIR_NAME = "images"
VIDEOS_SUBDIR_NAME = "videos"


def ensure_local_media_subdirs(base_dir: str | Path) -> tuple[Path, Path]:
    """Create (if missing) the `images/` and `videos/` subfolders under
    `base_dir`. Returns `(images_dir, videos_dir)`. Safe to call
    repeatedly -- never touches any files that already exist there.

    Raises whatever `Path.mkdir()` would raise (e.g. permission denied)
    -- unlike `user_media.py`'s dedicated per-deity folders (which are
    created automatically at app startup, so failures there are silently
    logged), this is called right after the user actively picks a folder
    in a file dialog, so a real error here (e.g. picked a read-only
    location) should surface to the user immediately rather than fail
    silently.
    """
    base = Path(base_dir)
    images_dir = base / IMAGES_SUBDIR_NAME
    videos_dir = base / VIDEOS_SUBDIR_NAME
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, videos_dir


def _scan_subdir(folder: Path, kind: str, exts: set[str]) -> list[UserMediaAsset]:
    assets: list[UserMediaAsset] = []
    if not folder.is_dir():
        return assets
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            assets.append(UserMediaAsset(kind=kind, local_path=str(path)))
    return assets


def scan_local_media_dir(base_dir: str | Path) -> list[UserMediaAsset]:
    """Return every usable image/video file found in
    `<base_dir>/images/` and `<base_dir>/videos/`.

    Non-recursive within each subfolder, never raises (a missing/
    inaccessible folder just yields an empty list for that subfolder --
    see this module's docstring for why an empty result here must NOT
    trigger a fallback to online sources in the caller).
    """
    base = Path(base_dir)
    try:
        images = _scan_subdir(base / IMAGES_SUBDIR_NAME, "image", _SUPPORTED_IMAGE_EXTS)
        videos = _scan_subdir(base / VIDEOS_SUBDIR_NAME, "video", _SUPPORTED_VIDEO_EXTS)
    except Exception:
        log.warning("Failed to scan local media folder %s", base, exc_info=True)
        return []

    assets = images + videos
    if assets:
        log.info(
            "Local media folder %s: found %d image(s), %d video(s)",
            base, len(images), len(videos),
        )
    return assets
