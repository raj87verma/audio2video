"""Main application window: three tabs (Create Video, Settings, About)."""
from __future__ import annotations

import os
import subprocess
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import RESOLUTION_CHOICES, WHISPER_MODEL_CHOICES, Settings
from .worker import PipelineWorker

AUDIO_FILE_FILTER = "Audio files (*.mp3 *.wav *.flac *.m4a *.ogg *.aac *.wma);;All files (*.*)"


class CreateVideoTab(QWidget):
    def __init__(self, get_settings, parent=None):
        super().__init__(parent)
        self._get_settings = get_settings
        self._audio_path: str | None = None
        self._worker: PipelineWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- File picker row -------------------------------------------------
        file_row = QHBoxLayout()
        self.file_label = QLabel("No audio file selected.")
        self.file_label.setWordWrap(True)
        browse_btn = QPushButton("Choose Audio File...")
        browse_btn.clicked.connect(self._choose_file)
        file_row.addWidget(browse_btn)
        file_row.addWidget(self.file_label, stretch=1)
        layout.addLayout(file_row)

        # --- Output path row --------------------------------------------------
        out_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Leave empty to auto-generate an output filename")
        out_browse_btn = QPushButton("Save As...")
        out_browse_btn.clicked.connect(self._choose_output)
        out_row.addWidget(QLabel("Output:"))
        out_row.addWidget(self.output_edit, stretch=1)
        out_row.addWidget(out_browse_btn)
        layout.addLayout(out_row)

        # --- Action buttons ------------------------------------------------
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Create Video")
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        self.open_btn = QPushButton("Open Output Folder")
        self.open_btn.clicked.connect(self._open_output_folder)
        self.open_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.open_btn)
        layout.addLayout(btn_row)

        # --- Progress --------------------------------------------------------
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready.")
        layout.addWidget(self.status_label)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        layout.addWidget(self.log_view, stretch=1)

        self._last_output_path: str | None = None

    # -- file pickers ---------------------------------------------------------
    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose an audio file", "", AUDIO_FILE_FILTER)
        if path:
            self._audio_path = path
            self.file_label.setText(path)

    def _choose_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save video as", "", "MP4 video (*.mp4)")
        if path:
            if not path.lower().endswith(".mp4"):
                path += ".mp4"
            self.output_edit.setText(path)

    def _open_output_folder(self) -> None:
        if not self._last_output_path:
            return
        folder = os.path.dirname(self._last_output_path)
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", folder])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            elif sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore[attr-defined]
        except Exception:
            QMessageBox.information(self, "Output location", f"Video saved at:\n{self._last_output_path}")

    # -- pipeline lifecycle -----------------------------------------------
    def _start(self) -> None:
        if not self._audio_path:
            QMessageBox.warning(self, "No audio file", "Please choose an audio file first.")
            return
        if self._worker is not None and self._worker.isRunning():
            return

        settings = self._get_settings()
        output_path = self.output_edit.text().strip() or None

        self.log_view.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.open_btn.setEnabled(False)

        self._worker = PipelineWorker(self._audio_path, settings, output_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished_ok)
        self._worker.finished_error.connect(self._on_finished_error)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
            self.status_label.setText("Cancelling...")

    def _on_progress(self, fraction: float, message: str) -> None:
        self.progress_bar.setValue(int(fraction * 100))
        self.status_label.setText(message)
        self.log_view.appendPlainText(f"[{fraction * 100:5.1f}%] {message}")

    def _on_finished_ok(self, result) -> None:
        self._last_output_path = result.output_path
        self.status_label.setText(f"Done! Mood: {result.mood.label} — saved to {result.output_path}")
        self.log_view.appendPlainText(
            f"\nFinished. Mood='{result.mood.label}', shots={len(result.shots)}, "
            f"stock_media={result.used_stock_media_count}, procedural={result.used_procedural_count}, "
            f"render_time={result.render_seconds:.1f}s"
        )
        self._reset_buttons(finished=True)

    def _on_finished_error(self, message: str) -> None:
        self.status_label.setText("Failed.")
        self.log_view.appendPlainText(f"\nERROR: {message}")
        QMessageBox.critical(self, "Video creation failed", message)
        self._reset_buttons(finished=False)

    def _on_cancelled(self) -> None:
        self.status_label.setText("Cancelled.")
        self.log_view.appendPlainText("\nCancelled by user.")
        self._reset_buttons(finished=False)

    def _reset_buttons(self, finished: bool) -> None:
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.open_btn.setEnabled(finished)


class SettingsTab(QWidget):
    def __init__(self, settings: Settings, on_save, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._on_save = on_save
        self._build_ui()
        self._load_from_settings()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        keys_group = QGroupBox("Free Stock Media & SFX API Keys (all optional)")
        form = QFormLayout()
        self.pexels_edit = QLineEdit()
        self.pexels_edit.setEchoMode(QLineEdit.Password)
        self.pixabay_edit = QLineEdit()
        self.pixabay_edit.setEchoMode(QLineEdit.Password)
        self.freesound_edit = QLineEdit()
        self.freesound_edit.setEchoMode(QLineEdit.Password)
        form.addRow("Pexels API key:", self.pexels_edit)
        form.addRow("Pixabay API key:", self.pixabay_edit)
        form.addRow("Freesound API key:", self.freesound_edit)
        keys_group.setLayout(form)
        layout.addWidget(keys_group)

        note = QLabel(
            "Leave any key blank to skip that provider — the app will still\n"
            "produce a complete video using procedurally generated visuals\n"
            "and synthetic sound effects. See the About tab for free sign-up links."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        video_group = QGroupBox("Video Settings")
        vform = QFormLayout()
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(RESOLUTION_CHOICES)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(10, 60)
        self.prefer_video_check = QCheckBox("Prefer stock video clips over still images when available")
        vform.addRow("Resolution:", self.resolution_combo)
        vform.addRow("Frame rate (fps):", self.fps_spin)
        vform.addRow(self.prefer_video_check)
        video_group.setLayout(vform)
        layout.addWidget(video_group)

        speech_group = QGroupBox("Lyrics / Speech Detection (runs fully offline, first use downloads a small model)")
        sform = QFormLayout()
        self.use_stt_check = QCheckBox("Detect lyrics/speech to guide visual keywords")
        self.whisper_combo = QComboBox()
        self.whisper_combo.addItems(WHISPER_MODEL_CHOICES)
        sform.addRow(self.use_stt_check)
        sform.addRow("Whisper model size:", self.whisper_combo)
        speech_group.setLayout(sform)
        layout.addWidget(speech_group)

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)
        layout.addStretch(1)

    def _load_from_settings(self) -> None:
        s = self.settings
        self.pexels_edit.setText(s.pexels_api_key)
        self.pixabay_edit.setText(s.pixabay_api_key)
        self.freesound_edit.setText(s.freesound_api_key)
        self.resolution_combo.setCurrentText(s.target_resolution)
        self.fps_spin.setValue(s.fps)
        self.prefer_video_check.setChecked(s.prefer_video_clips)
        self.use_stt_check.setChecked(s.use_speech_to_text)
        self.whisper_combo.setCurrentText(s.whisper_model_size)

    def _save(self) -> None:
        s = self.settings
        s.pexels_api_key = self.pexels_edit.text().strip()
        s.pixabay_api_key = self.pixabay_edit.text().strip()
        s.freesound_api_key = self.freesound_edit.text().strip()
        s.target_resolution = self.resolution_combo.currentText()
        s.fps = self.fps_spin.value()
        s.prefer_video_clips = self.prefer_video_check.isChecked()
        s.use_speech_to_text = self.use_stt_check.isChecked()
        s.whisper_model_size = self.whisper_combo.currentText()
        s.save()
        self._on_save(s)
        QMessageBox.information(self, "Settings saved", "Your settings have been saved.")


ABOUT_HTML = """
<h2>Audio2Video</h2>
<p>Turns an audio file into an automatically edited video using
<b>free and open resources only</b>:</p>
<ul>
  <li><b>Audio analysis</b> — tempo, beats, energy, mood (librosa, runs locally)</li>
  <li><b>Lyrics/speech detection</b> — faster-whisper, runs fully offline</li>
  <li><b>Stock visuals</b> — <a href="https://www.pexels.com/api/">Pexels</a> and
      <a href="https://pixabay.com/api/docs/">Pixabay</a> free APIs (free sign-up, no credit card)</li>
  <li><b>Procedural fallback visuals</b> — generated locally, used automatically whenever
      no API key is configured or no stock result matches</li>
  <li><b>Sound effects</b> — synthetic (numpy-generated) by default, optionally
      real keyword-matched effects via <a href="https://freesound.org/apiv2/apply/">Freesound</a>
      (free API token)</li>
  <li><b>Rendering</b> — MoviePy + FFmpeg</li>
</ul>
<p><b>Getting free API keys (optional, but improves visual variety):</b></p>
<ol>
  <li>Pexels: sign up at pexels.com and request a free API key at
      <a href="https://www.pexels.com/api/">pexels.com/api</a></li>
  <li>Pixabay: sign up at pixabay.com and get your key at
      <a href="https://pixabay.com/api/docs/">pixabay.com/api/docs</a></li>
  <li>Freesound (optional, for real SFX): apply at
      <a href="https://freesound.org/apiv2/apply/">freesound.org/apiv2/apply</a></li>
</ol>
<p>Paste the keys into the <b>Settings</b> tab. The app works completely without any
of them — it just relies more on generated visuals and synthetic sound effects.</p>
<p><b>License note:</b> Pexels and Pixabay content is free to use per their own
license terms; always check the current license text on their sites before
redistributing generated videos commercially.</p>
"""


class AboutTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(ABOUT_HTML)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        label.setTextFormat(Qt.RichText)
        layout.addWidget(label)
        layout.addStretch(1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio2Video — Free Audio-to-Video Generator")
        self.resize(900, 700)

        self.settings = Settings.load()

        tabs = QTabWidget()
        self.create_tab = CreateVideoTab(get_settings=lambda: self.settings)
        self.settings_tab = SettingsTab(self.settings, on_save=self._on_settings_saved)
        self.about_tab = AboutTab()

        tabs.addTab(self.create_tab, "Create Video")
        tabs.addTab(self.settings_tab, "Settings")
        tabs.addTab(self.about_tab, "About / Help")

        self.setCentralWidget(tabs)

    def _on_settings_saved(self, settings: Settings) -> None:
        self.settings = settings
