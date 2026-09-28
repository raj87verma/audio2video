"""PyInstaller / standalone entry point for Audio2Video.

`app/main.py` uses package-relative imports (``from .gui.main_window import
...``), so it must be launched as part of the ``app`` package rather than
executed directly as a loose script. This thin wrapper at the repo root is
what PyInstaller's ``--onedir``/``--onefile`` build points at (see
``packaging/audio2video.spec`` and the GitHub Actions workflow), and it is
also a convenient way to run the app from source without remembering the
``python -m app.main`` invocation:

    python run.py

Frozen (PyInstaller) execution additionally needs multiprocessing's
``freeze_support()`` guard — some of our dependencies (e.g. libraries used
transitively by numpy/onnxruntime for faster-whisper) can spawn worker
processes, and without this guard a frozen Windows .exe would re-launch
itself recursively.
"""
from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    from app.main import main as app_main

    return app_main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
