"""Main application window: three tabs (Create Video, Settings, About)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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
from ..core.local_media import ensure_local_media_subdirs, scan_local_media_dir
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
            f"local_media={result.used_local_media_count}, procedural={result.used_procedural_count}, "
            f"render_time={result.render_seconds:.1f}s"
        )
        if result.used_local_media_count:
            self.log_view.appendPlainText(
                f"\nUsed your local media folder for {result.used_local_media_count} shot(s) in this video."
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

        local_media_group = QGroupBox(
            "Media Folder (your own photos/videos — the ONLY visual source Audio2Video uses)"
        )
        lmform = QFormLayout()
        self.local_media_path_edit = QLineEdit()
        self.local_media_path_edit.setPlaceholderText("No folder selected — videos will use generated visuals")
        self.local_media_path_edit.editingFinished.connect(self._on_local_media_path_edited)
        local_browse_btn = QPushButton("Browse...")
        local_browse_btn.clicked.connect(self._browse_local_media_folder)
        local_clear_btn = QPushButton("Clear")
        local_clear_btn.clicked.connect(self._clear_local_media_folder)
        local_open_btn = QPushButton("Open Folder")
        local_open_btn.clicked.connect(self._open_local_media_folder)
        local_path_row = QHBoxLayout()
        local_path_row.addWidget(self.local_media_path_edit, stretch=1)
        local_path_row.addWidget(local_browse_btn)
        local_path_row.addWidget(local_clear_btn)
        local_path_row.addWidget(local_open_btn)
        lmform.addRow("Folder:", local_path_row)
        self.local_media_count_label = QLabel("")
        self.local_media_count_label.setWordWrap(True)
        lmform.addRow(self.local_media_count_label)
        local_media_note = QLabel(
            "Audio2Video does not use Pexels, Pixabay, Wikimedia Commons, or any other\n"
            "online resource for visuals — it never has an option to. The only source\n"
            "of real photos/videos is a folder you choose here. Two subfolders —\n"
            "\"images\" and \"videos\" — are created automatically the first time you\n"
            "pick a folder below; you can put files in those, or just drop them\n"
            "directly in the main folder itself — both are scanned.\n\n"
            "Every shot in every video you create cycles through the files in this\n"
            "folder (images and videos mixed together). If the folder is empty or no\n"
            "folder is set, shots use an animated generated background instead — this\n"
            "never changes based on the song, since there is no online source to fall\n"
            "back to.\n\n"
            "Videos are used in short, randomly varied 2-4 second excerpts per shot —\n"
            "never the whole clip at once — looping that excerpt if a shot needs more\n"
            "than 4 seconds. Photos get an automatic animated glow + twinkling sparkle\n"
            "effect added on top of the usual pan/zoom, so they don't look static.\n\n"
            "Click \"Clear\" to unset the folder and use only generated visuals."
        )
        local_media_note.setWordWrap(True)
        lmform.addRow(local_media_note)
        local_media_group.setLayout(lmform)
        layout.addWidget(local_media_group)

        video_group = QGroupBox("Video Settings")
        vform = QFormLayout()
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(RESOLUTION_CHOICES)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(10, 60)
        vform.addRow("Resolution:", self.resolution_combo)
        vform.addRow("Frame rate (fps):", self.fps_spin)
        video_group.setLayout(vform)
        layout.addWidget(video_group)

        speech_group = QGroupBox("Lyrics / Speech Detection (runs fully offline, first use downloads a small model)")
        sform = QFormLayout()
        self.use_stt_check = QCheckBox("Detect lyrics/speech (used only to pace devotional content, see About tab)")
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
        self.resolution_combo.setCurrentText(s.target_resolution)
        self.fps_spin.setValue(s.fps)
        self.use_stt_check.setChecked(s.use_speech_to_text)
        self.whisper_combo.setCurrentText(s.whisper_model_size)
        self.local_media_path_edit.setText(s.local_media_dir)
        self._refresh_local_media_count()

    def _persist_local_media_dir(self) -> None:
        """Write the folder currently shown in the text field straight into
        the shared Settings object (and disk) immediately.

        This used to happen ONLY inside `_save()`, triggered exclusively by
        the "Save Settings" button. That meant Browse-ing to a folder (and
        seeing "Found N video(s)") did NOT actually make that folder usable
        by "Create Video" unless the user remembered to also click "Save
        Settings" first. Clicking Create Video in between left the pipeline
        running against the old/empty `local_media_dir`, silently falling
        back to generated visuals. Persisting on every change (Browse,
        Clear, and manual edits) removes that trap entirely.
        """
        path = self.local_media_path_edit.text().strip()
        if self.settings.local_media_dir == path:
            return
        self.settings.local_media_dir = path
        self.settings.save()

    def _on_local_media_path_edited(self) -> None:
        self._persist_local_media_dir()
        self._refresh_local_media_count()

    def _refresh_local_media_count(self) -> None:
        """Re-scan the currently-entered local-media path and update the
        count label. Called after loading settings, after Browse/Clear,
        and after Save, so the counts shown are never stale relative to
        whatever's actually on disk or in the text field.
        """
        path = self.local_media_path_edit.text().strip()
        if not path:
            self.local_media_count_label.setText("")
            return
        assets = scan_local_media_dir(path)
        images = sum(1 for a in assets if a.kind == "image")
        videos = sum(1 for a in assets if a.kind == "video")
        if assets:
            self.local_media_count_label.setText(
                f"Found {images} image(s) and {videos} video(s) in this folder."
            )
        else:
            self.local_media_count_label.setText(
                "This folder currently has no usable images/videos in it — "
                "shots will use generated visuals until you add some."
            )

    def _browse_local_media_folder(self) -> None:
        start_dir = self.local_media_path_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Choose your media folder", start_dir)
        if not path:
            return
        try:
            ensure_local_media_subdirs(path)
        except Exception as exc:
            QMessageBox.critical(
                self, "Could not use this folder",
                f"Couldn't create 'images'/'videos' subfolders in:\n{path}\n\n{exc}",
            )
            return
        self.local_media_path_edit.setText(path)
        self._persist_local_media_dir()
        self._refresh_local_media_count()

    def _clear_local_media_folder(self) -> None:
        self.local_media_path_edit.clear()
        self._persist_local_media_dir()
        self._refresh_local_media_count()

    def _open_local_media_folder(self) -> None:
        folder = self.local_media_path_edit.text().strip()
        if not folder:
            QMessageBox.information(self, "No folder set", "Choose a folder with Browse... first.")
            return
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", folder])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            elif sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore[attr-defined]
        except Exception:
            QMessageBox.information(self, "Folder location", f"Your media folder is at:\n{folder}")

    def _save(self) -> None:
        s = self.settings
        s.target_resolution = self.resolution_combo.currentText()
        s.fps = self.fps_spin.value()
        s.use_speech_to_text = self.use_stt_check.isChecked()
        s.whisper_model_size = self.whisper_combo.currentText()
        s.local_media_dir = self.local_media_path_edit.text().strip()
        s.save()
        self._refresh_local_media_count()
        self._on_save(s)
        QMessageBox.information(self, "Settings saved", "Your settings have been saved.")


ABOUT_HTML = """
<h2>Audio2Video</h2>
<p>Turns an audio file into an automatically edited video using
<b>only local, free, and open resources — no online media of any kind</b>:</p>
<ul>
  <li><b>Audio analysis</b> — tempo, beats, energy, mood (librosa, runs locally)</li>
  <li><b>Lyrics/speech detection</b> — faster-whisper, runs fully offline</li>
  <li><b>Your own media folder</b> — the only source of real photos/videos.
      Point the <b>Settings</b> tab at a folder of your own royalty-free photos/videos
      and every shot in every video cycles through it. Videos are shown in short,
      randomly varied 2-4 second excerpts; photos get an automatic animated
      glow/sparkle effect added</li>
  <li><b>Procedural generated visuals</b> — animated gradient/particle/waveform
      backgrounds, generated locally, used automatically whenever no media folder
      is configured or it's empty</li>
  <li><b>Sound effects</b> — synthetic, numpy-generated whooshes/impacts/risers/sparkles,
      beat-synced to your track</li>
  <li><b>Rendering</b> — MoviePy + FFmpeg</li>
</ul>
<p>Audio2Video has <b>no settings, options, or code paths that contact Pexels,
Pixabay, Wikimedia Commons, Freesound, or any other online service</b> — every
visual and sound effect is either something you supplied yourself in a local
folder, or generated on your own machine.</p>
<p><b>Content-aware pacing:</b> if a song's filename or lyrics suggest devotional/
spiritual content (aarti, bhajan, kirtan, a deity's name, ...), shots are made
longer and slower to match typical contemplative pacing — this only affects
timing, not which visuals are used.</p>
<p><b>Licensing:</b> only use media you actually have the rights to use in your
media folder (your own photos/videos, or ones whose license explicitly permits
this kind of use) — Audio2Video doesn't check licensing for you, since every
file in that folder is something you chose to put there.</p>
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
