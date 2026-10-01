"""Lets a user point Audio2Video at a folder of their own royalty-free
photos/videos -- the ONLY source of real photos/videos this app has.

Audio2Video has no online media sources of any kind (no Pexels, no
Pixabay, no Wikimedia Commons, nothing) -- every shot's visual either
comes from this local folder, or is procedurally generated
(`procedural_visuals.py`).

Folder layout: a single path (stored in `Settings.local_media_dir`), with
two fixed subfolders:

    <local_media_dir>/
      images/    <- any number of photos
      videos/    <- any number of video clips

`ensure_local_media_subdirs()` creates these two subfolders under
whatever path the user has chosen (idempotent, safe to call repeatedly).
`scan_local_media_dir()` returns every usable image/video found in them.
An empty or unset folder simply means every shot falls back to the
procedural generator (see `pipeline.py`) -- there is no other source to
fall back to.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

IMAGES_SUBDIR_NAME = "images"
VIDEOS_SUBDIR_NAME = "videos"

_SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


@dataclass
class LocalMediaAsset:
    kind: str    # 'image' or 'video'
    local_path: str


def ensure_local_media_subdirs(base_dir: str | Path) -> tuple[Path, Path]:
    """Create (if missing) the `images/` and `videos/` subfolders under
    `base_dir`. Returns `(images_dir, videos_dir)`. Safe to call
    repeatedly -- never touches any files that already exist there.

    Raises whatever `Path.mkdir()` would raise (e.g. permission denied)
    -- this is called right after the user actively picks a folder in a
    file dialog, so a real error here (e.g. picked a read-only location)
    should surface to the user immediately rather than fail silently.
    """
    base = Path(base_dir)
    images_dir = base / IMAGES_SUBDIR_NAME
    videos_dir = base / VIDEOS_SUBDIR_NAME
    images_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, videos_dir


def _scan_subdir(folder: Path, kind: str, exts: set[str]) -> list[LocalMediaAsset]:
    assets: list[LocalMediaAsset] = []
    if not folder.is_dir():
        return assets
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            assets.append(LocalMediaAsset(kind=kind, local_path=str(path)))
    return assets


def scan_local_media_dir(base_dir: str | Path) -> list[LocalMediaAsset]:
    """Return every usable image/video file found in `<base_dir>/images/`
    and `<base_dir>/videos/`, PLUS any usable image/video files placed
    directly inside `<base_dir>` itself (classified automatically by
    file extension).

    Why the base_dir fallback exists: a real user, after being told "pick
    any folder, Audio2Video will create images/videos subfolders inside
    it", picked an existing folder that already had their photos/videos
    sitting directly in it and dropped files straight there rather than
    into the two newly-created (and, at a glance in a file browser,
    easy-to-miss-among-dozens-of-other-files) empty subfolders -- so the
    folder genuinely had their Khatu Shyam footage in it, but the scan
    (which at the time only looked inside images/ and videos/) found
    nothing, and every shot silently fell back to the generated orange
    gradient background instead -- confirmed via a real screenshot
    showing exactly that file layout and exactly that fallback output.
    Rather than only fix the in-app wording, this makes the scan itself
    tolerant of files left directly in the chosen folder: anything with a
    recognized image/video extension sitting right in `base_dir` (not
    inside any further subfolder) is picked up automatically, in
    addition to (not instead of) the dedicated images/videos subfolders.

    Non-recursive beyond that one extra level, never raises (a missing/
    inaccessible folder just yields an empty list).
    """
    base = Path(base_dir)
    try:
        images = _scan_subdir(base / IMAGES_SUBDIR_NAME, "image", _SUPPORTED_IMAGE_EXTS)
        videos = _scan_subdir(base / VIDEOS_SUBDIR_NAME, "video", _SUPPORTED_VIDEO_EXTS)

        # Files dropped directly in base_dir itself (not in a subfolder at
        # all) -- classify each by its own extension. Skip the images/
        # and videos/ subfolder entries themselves (they're directories,
        # `path.is_file()` already excludes them, but being explicit here
        # for clarity) and anything already counted above.
        loose_images: list[LocalMediaAsset] = []
        loose_videos: list[LocalMediaAsset] = []
        if base.is_dir():
            for path in sorted(base.iterdir()):
                if not path.is_file():
                    continue
                ext = path.suffix.lower()
                if ext in _SUPPORTED_IMAGE_EXTS:
                    loose_images.append(LocalMediaAsset(kind="image", local_path=str(path)))
                elif ext in _SUPPORTED_VIDEO_EXTS:
                    loose_videos.append(LocalMediaAsset(kind="video", local_path=str(path)))
    except Exception:
        log.warning("Failed to scan local media folder %s", base, exc_info=True)
        return []

    images = images + loose_images
    videos = videos + loose_videos
    assets = images + videos
    if assets:
        log.info(
            "Local media folder %s: found %d image(s), %d video(s) "
            "(including %d loose file(s) found directly in the folder, outside images/videos)",
            base, len(images), len(videos), len(loose_images) + len(loose_videos),
        )
    return assets
