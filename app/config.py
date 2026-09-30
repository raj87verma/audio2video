"""Central configuration: paths, cache locations and API keys.

All third-party services used here have a FREE tier that only requires a
free sign-up (no credit card):

  - Pexels    : https://www.pexels.com/api/            (free, generous limits)
  - Pixabay   : https://pixabay.com/api/docs/           (free, generous limits)
  - Freesound : https://freesound.org/docs/api/         (free, optional, for SFX)

If no keys are configured the app still works end-to-end: it falls back to
procedurally generated visuals (gradient/particle backgrounds synced to the
music) and synthetic, numpy-generated sound effects, so nothing ever hard
fails just because a key is missing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("AUDIO2VIDEO_HOME", Path.home() / ".audio2video"))
CACHE_DIR = DATA_DIR / "cache"
MEDIA_CACHE_DIR = CACHE_DIR / "media"
SFX_CACHE_DIR = CACHE_DIR / "sfx"
OUTPUT_DIR = DATA_DIR / "output"
SETTINGS_FILE = DATA_DIR / "settings.json"
# Where a user can drop their own royalty-free deity photos/videos (e.g.
# their own Khatu Shyam clips) so the app uses those FIRST -- ahead of
# Wikimedia/Pexels/Pixabay -- for every shot in a song about that deity.
# See user_media.py for the per-deity subfolder layout this contains.
USER_MEDIA_DIR = DATA_DIR / "user_media"

for _d in (DATA_DIR, CACHE_DIR, MEDIA_CACHE_DIR, SFX_CACHE_DIR, OUTPUT_DIR, USER_MEDIA_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    pexels_api_key: str = ""
    pixabay_api_key: str = ""
    freesound_api_key: str = ""
    whisper_model_size: str = "base"  # tiny/base/small/medium
    use_speech_to_text: bool = True
    target_resolution: str = "1080p"  # 720p / 1080p / 4k
    fps: int = 30
    prefer_video_clips: bool = True  # prefer stock video over still images when available
    # Wikimedia Commons needs no API key/sign-up (unlike Pexels/Pixabay)
    # and is only ever consulted for a *specific* detected deity/entity
    # (see content_hints.py) -- generic devotional/mood keywords still go
    # to Pexels/Pixabay as usual. Defaults on since it requires no setup
    # and only activates for a narrow, deliberately-targeted query type;
    # can be turned off for users who'd rather skip an extra network
    # source or who have attribution concerns (Wikimedia media is
    # typically CC BY-SA, which requires crediting the author -- see the
    # auto-generated credits file this produces).
    use_wikimedia: bool = True
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Settings":
        if SETTINGS_FILE.exists():
            try:
                data = json.loads(SETTINGS_FILE.read_text())
                known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                return cls(**known)
            except Exception:
                pass
        # Fall back to environment variables (useful for CLI / CI usage)
        return cls(
            pexels_api_key=os.environ.get("PEXELS_API_KEY", ""),
            pixabay_api_key=os.environ.get("PIXABAY_API_KEY", ""),
            freesound_api_key=os.environ.get("FREESOUND_API_KEY", ""),
        )

    def save(self) -> None:
        SETTINGS_FILE.write_text(json.dumps(asdict(self), indent=2))

    @property
    def resolution_px(self) -> tuple[int, int]:
        return {
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "4k": (3840, 2160),
        }.get(self.target_resolution, (1920, 1080))


RESOLUTION_CHOICES = ["720p", "1080p", "4k"]
WHISPER_MODEL_CHOICES = ["tiny", "base", "small", "medium"]
