"""Shot planner: turns the audio timeline into a sequence of beat-aligned
"shots" — each one describing a time window, an energy level, search
keywords for stock media, and hints for camera movement (Ken Burns
zoom/pan) and highlight moments (for SFX/VFX flashes on strong transients).

This is the bridge between raw signal analysis (audio_analysis, mood,
transcribe) and the video builder: everything the video builder needs to
assemble a beat-synced edit lives on the Shot objects produced here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import cycle

import numpy as np

from .audio_analysis import AudioFeatures
from .mood import MoodProfile
from .transcribe import TranscriptResult

log = logging.getLogger(__name__)

MIN_SHOT_DURATION = 1.5   # seconds — even the fastest cuts stay watchable
MAX_SHOT_DURATION = 8.0   # seconds — even the slowest sections still cut occasionally

_ZOOM_DIRECTIONS = ["in", "out"]
_PAN_DIRECTIONS = ["left", "right", "up", "down", "none"]


@dataclass
class Shot:
    index: int
    start: float
    end: float
    energy_level: str            # 'low' / 'medium' / 'high'
    keywords: list[str]
    is_highlight: bool           # strong transient inside -> SFX/VFX flash cue
    highlight_time: float | None  # absolute time of the triggering onset, if any
    zoom_direction: str           # 'in' / 'out' for Ken Burns effect
    pan_direction: str            # 'left' / 'right' / 'up' / 'down' / 'none'
    prefer_video: bool            # True -> try to fetch a video clip; False -> still image is fine

    @property
    def duration(self) -> float:
        return self.end - self.start


def _energy_level(features: AudioFeatures, start: float, end: float, low_thr: float, high_thr: float) -> float:
    mask = (features.rms_times >= start) & (features.rms_times < end)
    if not np.any(mask):
        return float(features.energy_at((start + end) / 2))
    return float(np.mean(features.rms[mask]))


def _level_label(value: float, low_thr: float, high_thr: float) -> str:
    if value <= low_thr:
        return "low"
    if value >= high_thr:
        return "high"
    return "medium"


def _target_shot_duration(features: AudioFeatures, mood: MoodProfile) -> float:
    """Base shot length in seconds, derived from tempo and mood cut_speed."""
    if features.tempo and features.tempo > 0:
        beat_duration = 60.0 / features.tempo
    else:
        beat_duration = 0.5
    base = beat_duration * 4  # one bar (assuming 4/4) as the natural unit
    target = base * mood.cut_speed
    return float(np.clip(target, MIN_SHOT_DURATION, MAX_SHOT_DURATION))


def _boundaries_from_beats(features: AudioFeatures, target_duration: float) -> list[float]:
    """Build cut points snapped to detected beats, spaced ~target_duration apart."""
    beats = features.beat_times
    if beats is None or len(beats) < 4:
        return []

    boundaries = [0.0]
    next_target = target_duration
    for t in beats:
        if t >= next_target:
            boundaries.append(float(t))
            next_target = t + target_duration
    if boundaries[-1] < features.duration - 0.5:
        boundaries.append(features.duration)
    else:
        boundaries[-1] = features.duration
    return boundaries


def _boundaries_uniform(duration: float, target_duration: float) -> list[float]:
    n = max(1, round(duration / target_duration))
    step = duration / n
    return [round(i * step, 3) for i in range(n)] + [duration]


def plan_shots(
    features: AudioFeatures,
    mood: MoodProfile,
    transcript: TranscriptResult | None = None,
    prefer_video_clips: bool = True,
    content_keywords: list[str] | None = None,
    cut_speed_multiplier: float | None = None,
    prioritize_content_keywords: bool = False,
) -> list[Shot]:
    """Produce an ordered list of Shot objects covering the full audio duration.

    Cut points are snapped to detected beats when enough beats were found
    (typical for real music); otherwise falls back to uniform-length shots
    so the planner never fails even on beat-less/ambient/noisy audio.

    `content_keywords`, if given (see `content_hints.detect_content_hints`),
    take priority over the acoustic mood's own keywords for any shot that
    has no lyric-derived keywords of its own -- this lets a detected
    devotional/spiritual theme (or similar content-level signal) override
    generic tempo/loudness-based mood keywords like "celebration" or
    "dance" that would otherwise fetch visuals with no connection to what
    the song is actually about.

    `prioritize_content_keywords`, when True, flips that order so
    `content_keywords` are tried *before* lyric-derived keywords instead
    of after. This exists for devotional content specifically: real
    devotional lyrics (aarti/bhajan) are overwhelmingly invocation/
    grammar words -- "tera" (your), "karo" (do), "jai" (hail), "kalyan"
    (welfare), "bhakto" (devotees) -- which are not visual search terms
    and mean nothing to a stock-media API, unlike a pop song's lyrics
    which might genuinely describe a scene ("dancing in the rain"). Left
    at the default (lyric keywords first), a devotional song with clear
    vocals would have its correctly-detected deity/devotional keywords
    silently discarded in favor of these meaningless transcribed words
    for almost every shot, sending searches for "tera"/"karo" instead of
    "Khatu Shyam" -- defeating the entire purpose of content_hints'
    devotional/deity detection. Non-devotional callers should leave this
    False so genuinely useful lyric-derived keywords keep taking priority
    as before.
    """
    target_duration = _target_shot_duration(features, mood)
    if cut_speed_multiplier:
        target_duration = float(np.clip(target_duration * cut_speed_multiplier, MIN_SHOT_DURATION, MAX_SHOT_DURATION))

    boundaries = _boundaries_from_beats(features, target_duration)
    if len(boundaries) < 2:
        log.info("Not enough beats detected; falling back to uniform shot lengths")
        boundaries = _boundaries_uniform(features.duration, target_duration)

    low_thr = float(np.percentile(features.rms, 33)) if len(features.rms) else 0.33
    high_thr = float(np.percentile(features.rms, 66)) if len(features.rms) else 0.66

    onset_times = features.onset_times if features.onset_times is not None else np.array([])
    mood_keyword_cycle = cycle(mood.keywords) if mood.keywords else cycle(["abstract background"])
    content_keyword_cycle = cycle(content_keywords) if content_keywords else None
    zoom_cycle = cycle(_ZOOM_DIRECTIONS)
    pan_cycle = cycle(_PAN_DIRECTIONS)

    shots: list[Shot] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start < 0.05:
            continue

        energy_value = _energy_level(features, start, end, low_thr, high_thr)
        energy_label = _level_label(energy_value, low_thr, high_thr)

        # Highlight detection: is there a strong onset inside this shot?
        in_window = onset_times[(onset_times >= start) & (onset_times < end)]
        is_highlight = False
        highlight_time = None
        if len(in_window) > 0 and energy_label == "high":
            highlight_time = float(in_window[0])
            is_highlight = True

        # Keywords: normally prefer lyric/vocal keywords for this window,
        # then a content-hint override (e.g. devotional theme detected
        # from the filename), then fall back to cycling through the
        # mood's keyword list for visual variety. When
        # `prioritize_content_keywords` is set (devotional content -- see
        # this function's docstring), content keywords are tried first
        # instead, since transcribed devotional lyrics are almost never
        # useful visual search terms.
        keywords: list[str] = []
        transcript_keywords_fn = (
            (lambda: transcript.keywords_in(start, end, limit=3))
            if transcript is not None and transcript.has_speech
            else (lambda: [])
        )
        content_keywords_fn = (
            (lambda: [next(content_keyword_cycle)]) if content_keyword_cycle is not None else (lambda: [])
        )
        ordered_sources = (
            [content_keywords_fn, transcript_keywords_fn]
            if prioritize_content_keywords
            else [transcript_keywords_fn, content_keywords_fn]
        )
        for source in ordered_sources:
            keywords = source()
            if keywords:
                break
        if not keywords:
            keywords = [next(mood_keyword_cycle)]

        shots.append(
            Shot(
                index=i,
                start=round(start, 3),
                end=round(end, 3),
                energy_level=energy_label,
                keywords=keywords,
                is_highlight=is_highlight,
                highlight_time=highlight_time,
                zoom_direction=next(zoom_cycle),
                pan_direction=next(pan_cycle),
                prefer_video=prefer_video_clips,
            )
        )

    log.info(
        "Planned %d shots (avg %.2fs each, target=%.2fs) mood=%s",
        len(shots),
        features.duration / max(1, len(shots)),
        target_duration,
        mood.label,
    )
    return shots
