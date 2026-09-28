"""Audio2Video — turn an audio file into an auto-assembled video.

Built entirely on free / open-source tooling:
  - librosa            : audio analysis (tempo, beats, energy)
  - faster-whisper      : optional local speech-to-text (no API key, no cost)
  - Pexels / Pixabay    : free stock photo & video APIs (require free API keys)
  - Freesound           : optional free sound-effect API (requires free API key)
  - MoviePy / FFmpeg    : video assembly and rendering
  - PySide6             : desktop GUI
"""

__version__ = "0.1.0"

# --- Compatibility shim -----------------------------------------------------
# MoviePy 1.0.3 (the latest PyPI release as of writing) still references the
# long-deprecated `PIL.Image.ANTIALIAS` constant internally (in its resize
# effect). Modern Pillow (10+) removed that alias in favor of `LANCZOS`.
# Restoring the alias here — once, at import time — lets MoviePy's resize
# keep working without pinning to an old, less secure Pillow release.
from PIL import Image as _PILImage  # noqa: E402

if not hasattr(_PILImage, "ANTIALIAS"):
    _PILImage.ANTIALIAS = _PILImage.LANCZOS

