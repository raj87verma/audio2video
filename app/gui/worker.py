"""QThread worker that runs the audio->video pipeline in the background.

Keeps the Qt event loop responsive during the (potentially long) render by
running `run_pipeline` off the GUI thread, and bridges its progress
callback / cancel flag to Qt signals so the main window can update a
progress bar and log view safely.
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal

from ..config import DATA_DIR, Settings
from ..core.pipeline import PipelineCancelled, PipelineResult, run_pipeline

log = logging.getLogger(__name__)
LOG_FILE = DATA_DIR / "audio2video.log"


class PipelineWorker(QThread):
    progress = Signal(float, str)     # fraction 0..1, message
    finished_ok = Signal(object)      # PipelineResult
    finished_error = Signal(str)      # error message
    cancelled = Signal()

    def __init__(self, audio_path: str, settings: Settings, output_path: str | None = None, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.settings = settings
        self.output_path = output_path
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def _cancel_check(self) -> bool:
        return self._cancel_requested

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            result: PipelineResult = run_pipeline(
                self.audio_path,
                settings=self.settings,
                output_path=self.output_path,
                progress_cb=lambda frac, msg: self.progress.emit(frac, msg),
                cancel_check=self._cancel_check,
            )
            self.finished_ok.emit(result)
        except PipelineCancelled:
            self.cancelled.emit()
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            # str(exc) can be an EMPTY string for several real-world
            # exceptions (MemoryError, some OSError/subprocess failures,
            # etc.), which previously showed the user a bare "ERROR:" with
            # no actual information. Always include the exception's class
            # name so there's something readable even when the message
            # body is blank, and log the full traceback (visible in the
            # app's log file / console) for real debugging.
            message = str(exc).strip()
            detail = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
            log.error("Pipeline failed: %s", detail, exc_info=True)
            self.finished_error.emit(f"{detail}\n\nFull details logged to:\n{LOG_FILE}")
