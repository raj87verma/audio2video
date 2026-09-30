"""Central configuration: paths and settings.

Audio2Video uses NO online media sources of any kind for visuals -- every
shot's visual either comes from a folder of the user's own photos/videos
(see `local_media.py`), or is procedurally generated locally
(`procedural_visuals.py`). Nothing here ever calls out to the internet
for images or video.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("AUDIO2VIDEO_HOME", Path.home() / ".audio2video"))
OUTPUT_DIR = DATA_DIR / "output"
SETTINGS_FILE = DATA_DIR / "settings.json"

for _d in (DATA_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    whisper_model_size: str = "base"  # tiny/base/small/medium
    use_speech_to_text: bool = True
    target_resolution: str = "1080p"  # 720p / 1080p / 4k
    fps: int = 30
    # Absolute path to a folder (containing `images/` and `videos/`
    # subfolders -- see local_media.py) of the user's own photos/videos.
    # Empty string means "not set" -- every shot then uses procedurally
    # generated visuals instead. This is the ONLY visual media source
    # Audio2Video has; there is no online fallback of any kind.
    local_media_dir: str = ""

    @classmethod
    def load(cls) -> "Settings":
        if SETTINGS_FILE.exists():
            try:
                data = json.loads(SETTINGS_FILE.read_text())
                known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                return cls(**known)
            except Exception:
                pass
        return cls()

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
