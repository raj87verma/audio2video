"""Audio analysis: tempo, beats, energy envelope, spectral features.

Everything here uses librosa (open-source, free, runs fully offline — no
API keys, no network calls). The results feed the mood classifier and the
shot planner so the final video's cuts, zooms and color grading follow the
music instead of being arbitrary.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import librosa
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    path: str
    duration: float                 # seconds
    sample_rate: int
    tempo: float                    # BPM
    beat_times: np.ndarray          # seconds, one entry per detected beat
    downbeat_times: np.ndarray      # seconds, coarser subset used for hard cuts
    onset_times: np.ndarray         # seconds, transient/percussive hits (good for SFX cues)
    rms: np.ndarray                 # frame-wise loudness envelope (normalized 0..1)
    rms_times: np.ndarray           # seconds, matches `rms`
    spectral_centroid_mean: float   # brightness indicator (Hz)
    spectral_bandwidth_mean: float
    zero_crossing_rate_mean: float  # noisiness / percussiveness indicator
    harmonic_ratio: float           # 0..1, fraction of energy that is harmonic vs percussive
    loudness_mean: float            # mean RMS (0..1)
    loudness_std: float             # RMS variance -> dynamic range indicator
    energy_segments: list = field(default_factory=list)  # list of (start, end, level) low/med/high

    def energy_at(self, t: float) -> float:
        """Interpolated 0..1 energy level at time t seconds."""
        if len(self.rms_times) == 0:
            return 0.5
        return float(np.interp(t, self.rms_times, self.rms))

    def beats_in(self, start: float, end: float) -> np.ndarray:
        return self.beat_times[(self.beat_times >= start) & (self.beat_times < end)]


def _energy_segments(rms_times: np.ndarray, rms: np.ndarray, n_segments: int = 1) -> list:
    """Bucket the track into coarse low/medium/high energy segments.

    Uses simple tertile thresholds on the smoothed RMS curve. Returned as a
    list of (start_time, end_time, level) where level is one of
    'low' / 'medium' / 'high'. Consecutive frames with the same level are
    merged so the shot planner gets a small, usable number of segments
    instead of one entry per audio frame.
    """
    if len(rms) == 0:
        return []
    low_thr = np.percentile(rms, 33)
    high_thr = np.percentile(rms, 66)

    def level_for(v: float) -> str:
        if v <= low_thr:
            return "low"
        if v >= high_thr:
            return "high"
        return "medium"

    levels = [level_for(v) for v in rms]
    segments = []
    seg_start = rms_times[0]
    seg_level = levels[0]
    for i in range(1, len(levels)):
        if levels[i] != seg_level:
            segments.append((seg_start, rms_times[i], seg_level))
            seg_start = rms_times[i]
            seg_level = levels[i]
    segments.append((seg_start, rms_times[-1], seg_level))
    return segments


def analyze_audio(path: str, sr: int | None = 22050) -> AudioFeatures:
    """Load and analyze an audio file, returning an AudioFeatures bundle.

    Supports any format librosa/soundfile can decode (mp3, wav, flac, ogg,
    m4a via audioread fallback, etc.).
    """
    log.info("Loading audio: %s", path)
    y, sample_rate = librosa.load(path, sr=sr, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sample_rate))

    # --- Tempo & beats -----------------------------------------------------
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sample_rate, units="frames")
    tempo = float(np.atleast_1d(tempo)[0]) if np.ndim(tempo) else float(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
    # Downbeats: keep every 4th beat as an approximate bar marker for hard cuts.
    downbeat_times = beat_times[::4] if len(beat_times) >= 4 else beat_times

    # --- Onsets (good triggers for percussive SFX / VFX flashes) ----------
    onset_frames = librosa.onset.onset_detect(y=y, sr=sample_rate, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)

    # --- Loudness / energy envelope -----------------------------------------
    hop_length = 512
    rms_raw = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms_raw)), sr=sample_rate, hop_length=hop_length)
    rms_max = float(rms_raw.max()) if rms_raw.max() > 0 else 1.0
    rms = np.clip(rms_raw / rms_max, 0.0, 1.0)

    # --- Spectral descriptors (mood classifier inputs) ----------------------
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate, hop_length=hop_length)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy = float(np.sum(y_harmonic ** 2))
    percussive_energy = float(np.sum(y_percussive ** 2))
    total_energy = harmonic_energy + percussive_energy
    harmonic_ratio = harmonic_energy / total_energy if total_energy > 0 else 0.5

    segments = _energy_segments(rms_times, rms)

    features = AudioFeatures(
        path=path,
        duration=duration,
        sample_rate=sample_rate,
        tempo=round(tempo, 1),
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        onset_times=onset_times,
        rms=rms,
        rms_times=rms_times,
        spectral_centroid_mean=float(np.mean(spectral_centroid)),
        spectral_bandwidth_mean=float(np.mean(spectral_bandwidth)),
        zero_crossing_rate_mean=float(np.mean(zcr)),
        harmonic_ratio=float(harmonic_ratio),
        loudness_mean=float(np.mean(rms)),
        loudness_std=float(np.std(rms)),
        energy_segments=segments,
    )
    log.info(
        "Audio analyzed: duration=%.1fs tempo=%.1f BPM beats=%d onsets=%d",
        duration, tempo, len(beat_times), len(onset_times),
    )
    return features
