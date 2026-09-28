"""QThread worker that runs the audio->video pipeline in the background.

Keeps the Qt event loop responsive during the (potentially long) render by
running `run_pipeline` off the GUI thread, and bridges its progress
callback / cancel flag to Qt signals so the main window can update a
progress bar and log view safely.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ..config import Settings
from ..core.pipeline import PipelineCancelled, PipelineResult, run_pipeline


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
            self.finished_error.emit(str(exc))
