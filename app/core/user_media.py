"""Lets a user supply their own royalty-free deity photos/videos, used
FIRST -- ahead of Wikimedia Commons, Pexels, and Pixabay -- for every shot
in a song about that deity.

Why this exists: even with the Wikimedia Commons integration (see
wikimedia_fetcher.py) and deity-aware keyword prioritization (see
shot_planner.py's `prioritize_content_keywords`), only the *subset* of
shots whose cycling keyword happens to land on a deity-specific term
(e.g. "Khatu Shyam") actually gets deity footage -- the shot_planner
keyword cycle still mixes in generic devotional terms ("hindu temple",
"temple diya lamp", ...) for visual variety across a long track, and
those shots keep using Pexels/Pixabay/procedural generic visuals. Real
user feedback was that this still produces a video where several shots
look unrelated to the actual subject.

The fix here is simple and direct: if a user has supplied their own
photos/videos of the actual deity a song is about, there's no reason to
ever fall back to anything else for that song -- every single shot can
just use the user's own footage instead of only the deity-keyword shots.
This module provides the folder convention and scanning logic;
pipeline.py is responsible for using its result to bypass the normal
per-shot media resolver entirely when it finds anything.

Folder layout (all under `config.USER_MEDIA_DIR`, i.e.
`~/.audio2video/user_media/` by default, or `%AUDIO2VIDEO_HOME%\\user_media`
if that environment variable is set):

    user_media/
      khatu_shyam/      <- drop Khatu Shyam photos/videos here
      krishna/
      hanuman/
      ... one folder per recognized deity (see content_hints.py) ...

`ensure_user_media_dirs()` creates all of these up front (with a short
README.txt inside each) so a user has ready-made folders to drop files
into instead of needing to guess folder names or spellings.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import USER_MEDIA_DIR
from .content_hints import _DEITY_ALIASES

log = logging.getLogger(__name__)

_SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}

_README_TEMPLATE = """This folder is for your own royalty-free "{name}" photos/videos.

Drop any number of image files ({image_exts}) and/or video files
({video_exts}) directly in this folder (subfolders are not scanned).

When a song's filename or lyrics are recognized as being about "{name}",
Audio2Video will use ONLY the files you place here for every shot in
that video -- it will not fall back to Wikimedia Commons, Pexels, or
Pixabay for this deity as long as this folder has at least one usable
file in it. If this folder is empty, those online sources are used as
before.

Tips:
  - Video clips are used in short excerpts (a few seconds at a time) and
    a different, randomly chosen part of the clip is picked each time it
    reappears in a longer video, so a single video still gives visual
    variety across many shots.
  - Still images automatically get a subtle animated glow/sparkle effect
    added, so they don't look completely static in the final video.
  - Only use media you have the rights to use (your own photos/videos,
    or ones whose license explicitly allows this kind of use).

You can delete this README.txt file; it has no effect on how your media
is used.
"""

# All canonical deity names currently recognized by content_hints.py --
# derived directly from _DEITY_ALIASES so this list can never drift out
# of sync with what deity detection actually recognizes.
_CANONICAL_DEITY_NAMES: list[str] = sorted({name for name, _ in _DEITY_ALIASES.values()})


def slugify(name: str) -> str:
    """Turn a canonical deity display name into a filesystem-safe folder
    name, e.g. "Khatu Shyam" -> "khatu_shyam", "Sai Baba" -> "sai_baba".
    """
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def ensure_user_media_dirs() -> Path:
    """Create (if missing) one subfolder per recognized deity under
    `USER_MEDIA_DIR`, each with a short explanatory README.txt. Safe to
    call repeatedly (e.g. on every app launch) -- never overwrites an
    existing README.txt a user may have deleted on purpose, and never
    touches any media files.

    Returns `USER_MEDIA_DIR` itself, for convenience (e.g. so a caller
    can open it in a file browser right after ensuring it's populated).
    """
    for name in _CANONICAL_DEITY_NAMES:
        folder = USER_MEDIA_DIR / slugify(name)
        folder.mkdir(parents=True, exist_ok=True)
        readme = folder / "README.txt"
        if not readme.exists():
            try:
                readme.write_text(
                    _README_TEMPLATE.format(
                        name=name,
                        image_exts=", ".join(sorted(_SUPPORTED_IMAGE_EXTS)),
                        video_exts=", ".join(sorted(_SUPPORTED_VIDEO_EXTS)),
                    ),
                    encoding="utf-8",
                )
            except Exception:
                # Never let a README-writing hiccup (e.g. odd permissions)
                # block the app from starting or from using media that IS
                # there -- this file is purely a convenience for the user.
                log.warning("Could not write README.txt in %s", folder, exc_info=True)
    return USER_MEDIA_DIR


@dataclass
class UserMediaAsset:
    kind: str    # 'image' or 'video'
    local_path: str


def scan_user_media(deity_name: str) -> list[UserMediaAsset]:
    """Return every usable image/video file the user has placed in
    `USER_MEDIA_DIR/<slug(deity_name)>/`, or an empty list if that folder
    doesn't exist / is empty / has no recognized file types.

    Deliberately non-recursive (subfolders aren't scanned) and never
    raises -- a folder full of unrelated files (e.g. stray README edits,
    other document types) just yields whatever *is* a recognized image/
    video, silently ignoring the rest.
    """
    folder = USER_MEDIA_DIR / slugify(deity_name)
    if not folder.is_dir():
        return []

    assets: list[UserMediaAsset] = []
    try:
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext in _SUPPORTED_IMAGE_EXTS:
                assets.append(UserMediaAsset(kind="image", local_path=str(path)))
            elif ext in _SUPPORTED_VIDEO_EXTS:
                assets.append(UserMediaAsset(kind="video", local_path=str(path)))
    except Exception:
        log.warning("Failed to scan user media folder %s", folder, exc_info=True)
        return []

    if assets:
        log.info("Found %d user-supplied media file(s) for %r in %s", len(assets), deity_name, folder)
    return assets
