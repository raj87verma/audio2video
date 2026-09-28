"""Entry point for the Audio2Video desktop application."""
from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from .config import DATA_DIR
from .gui.main_window import MainWindow

LOG_FILE = DATA_DIR / "audio2video.log"


def _configure_logging() -> None:
    """Log to both a rotating-ish file and stderr.

    The Windows build is a windowed app (no console), so without a log
    file there is literally nowhere for anyone -- user or developer -- to
    see what happened when something fails. `LOG_FILE` lives right next to
    the app's settings/cache/output (`~/.audio2video/` by default, or
    `%AUDIO2VIDEO_HOME%` if set), so it's easy to point users at.
    """
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: list[logging.Handler] = []

    try:
        # Truncate on each launch rather than growing forever -- this app
        # only ever needs the *current* run's log for troubleshooting.
        file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt))
        handlers.append(file_handler)
    except Exception:
        # If the log file can't be created for some reason (e.g. a locked
        # file, unusual permissions), fall back to console-only logging
        # rather than crashing the app before it even starts.
        pass

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(fmt))
    handlers.append(stream_handler)

    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


def main() -> int:
    _configure_logging()
    logging.getLogger(__name__).info("Audio2Video starting up. Log file: %s", LOG_FILE)
    app = QApplication(sys.argv)
    app.setApplicationName("Audio2Video")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
