"""Top-level pipeline: audio file in, finished MP4 out.

Wires together every core module in order:

  1. audio_analysis.analyze_audio      — tempo, beats, energy, spectral features
  2. mood.classify_mood                — rule-based mood + keywords + color grade
  3. transcribe.transcribe_audio       — optional local speech-to-text (lyrics)
  4. shot_planner.plan_shots           — beat-aligned shot list
  5. local_media.scan_local_media_dir  — the user's own local photos/videos
     (procedural_visuals is used automatically as the video_builder's
     fallback whenever the local media folder is empty or unset)
  6. video_builder.assemble_video      — final render + SFX mix + export

Audio2Video uses NO online media sources of any kind for visuals -- every
shot's picture/video either comes from the user's own local media folder
(see `local_media.py` / `Settings.local_media_dir`) or is procedurally
generated (see `procedural_visuals.py`). There is nothing to configure
here that reaches the network for images/video, and nothing ever will,
by design.

Exposes a single `run_pipeline(...)` function with a `progress_cb` hook so
the GUI (or a CLI) can report fine-grained progress without needing to know
about any of the underlying modules.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import OUTPUT_DIR, Settings
from .audio_analysis import AudioFeatures, analyze_audio
from .content_hints import detect_content_hints
from .local_media import scan_local_media_dir
from .mood import MoodProfile, classify_mood
from .shot_planner import Shot, plan_shots
from .transcribe import TranscriptResult, transcribe_audio
from .video_builder import assemble_video

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


@dataclass
class PipelineResult:
    output_path: str
    mood: MoodProfile
    features: AudioFeatures
    transcript: TranscriptResult
    shots: list[Shot]
    render_seconds: float
    # Shots that used a file from the user's local media folder (see
    # local_media.py / Settings.local_media_dir).
    used_local_media_count: int = 0
    # Shots that had no local media available and used a procedurally
    # generated background instead (see procedural_visuals.py).
    used_procedural_count: int = 0


def _noop_progress(fraction: float, message: str) -> None:
    pass


def _default_output_path(audio_path: str) -> str:
    stem = Path(audio_path).stem
    ts = time.strftime("%Y%m%d_%H%M%S")
    return str(OUTPUT_DIR / f"{stem}_{ts}.mp4")


def _make_local_media_resolver(assets: list):
    """Build a `resolve() -> (local_path, kind)` closure that cycles
    through ALL of `assets` (images and videos together) round-robin.

    Every supplied file gets an equal turn in the rotation regardless of
    kind, so a folder containing both images and videos genuinely mixes
    both into the final video (an earlier version of this logic
    partitioned assets into separate image/video pools and always
    preferred video when any was present, which silently starved out any
    supplied photos -- confirmed via a real render showing 100% video
    content; fixed by removing that partitioning entirely).

    The rotation order is freshly shuffled every time this function is
    called (i.e. once per `run_pipeline()` call -- once per generated
    video). `scan_local_media_dir` always returns `assets` in a fixed,
    deterministic (alphabetical) order, and the round-robin index used
    to always start at 0 -- so every single video generated from the
    same media folder opened with the exact same file for shot #1 and
    then walked through the rest of the folder in the exact same
    alphabetical sequence every time, however many different songs were
    rendered. Confirmed via a real user's report: two different videos,
    generated from two different songs against the same media folder,
    both started on the identical clip. Shuffling a *copy* of `assets`
    here (never mutating the caller's list) means a new video gets a
    different starting clip and a different overall visitation order
    each time, while still guaranteeing -- via the same `% len(shuffled)`
    wraparound as before -- that every file in the folder gets an equal
    turn within that one video's rotation.

    Returned `kind` is `"local_image"` / `"local_video"` so
    `video_builder.build_shot_clip` applies the extra treatment meant for
    the user's own media: a glow/sparkle VFX overlay on stills, and a
    short randomized (2-4 second) trim for video excerpts.
    """
    if not assets:
        def resolve_empty() -> tuple[str | None, str]:
            return None, "procedural"
        return resolve_empty

    shuffled = assets.copy()
    random.shuffle(shuffled)

    index = {"i": 0}

    def resolve() -> tuple[str | None, str]:
        i = index["i"] % len(shuffled)
        index["i"] += 1
        asset = shuffled[i]
        return asset.local_path, f"local_{asset.kind}"

    return resolve


def run_pipeline(
    audio_path: str,
    settings: Settings | None = None,
    output_path: str | None = None,
    progress_cb: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> PipelineResult:
    """Run the full audio -> video pipeline and return a PipelineResult.

    `progress_cb(fraction, message)` is called repeatedly with fraction in
    [0, 1] and a short human-readable status string — safe to wire directly
    into a Qt progress bar / label.

    `cancel_check()` — if provided and returns True, the pipeline raises
    `PipelineCancelled` at the next safe checkpoint (between stages / between
    shots) instead of continuing.
    """
    settings = settings or Settings.load()
    progress_cb = progress_cb or _noop_progress
    output_path = output_path or _default_output_path(audio_path)

    def report(frac: float, msg: str) -> None:
        log.info("[%.0f%%] %s", frac * 100, msg)
        progress_cb(frac, msg)

    def check_cancel() -> None:
        if cancel_check and cancel_check():
            raise PipelineCancelled("Cancelled by user")

    # --- Stage 1: audio analysis (0% - 15%) ---------------------------------
    report(0.0, "Analyzing audio (tempo, beats, energy)...")
    features = analyze_audio(audio_path)
    check_cancel()

    # --- Stage 2: mood classification (15% - 18%) ---------------------------
    report(0.15, "Classifying mood...")
    mood = classify_mood(features)
    report(0.18, f"Mood detected: {mood.label}")
    check_cancel()

    # --- Stage 3: optional transcription (18% - 32%) ------------------------
    transcript = TranscriptResult(full_text="", has_speech=False)
    if settings.use_speech_to_text:
        report(0.20, "Transcribing vocals/lyrics (if any)...")
        transcript = transcribe_audio(audio_path, model_size=settings.whisper_model_size, enabled=True)
        if transcript.has_speech:
            report(0.32, "Lyrics detected — will use them to guide visuals.")
        else:
            report(0.32, "No lyrics detected — using mood-based visuals.")
    else:
        report(0.32, "Speech-to-text disabled — using mood-based visuals.")
    check_cancel()

    # --- Stage 4: shot planning (32% - 38%) ---------------------------------
    # Detect a devotional/spiritual theme from the filename plus any
    # transcribed lyrics, purely to slow shot pacing to something more
    # contemplative than the acoustic mood classifier alone would pick
    # (see content_hints.py's module docstring).
    content_hints = detect_content_hints(Path(audio_path).name, transcript.full_text)
    if content_hints.is_devotional:
        report(
            0.33,
            f"Devotional/spiritual theme detected ({', '.join(content_hints.matched_terms[:3])}) "
            "— using slower, more contemplative pacing.",
        )

    # Scan the user's local media folder now (before shot planning) purely
    # so the progress log can report whether it found anything before the
    # per-shot fetch loop starts.
    local_media_assets = scan_local_media_dir(settings.local_media_dir) if settings.local_media_dir else []
    if settings.local_media_dir:
        if local_media_assets:
            kinds_summary = (
                f"{sum(1 for a in local_media_assets if a.kind == 'video')} video(s), "
                f"{sum(1 for a in local_media_assets if a.kind == 'image')} photo(s)"
            )
            report(0.335, f"Using your local media folder ({kinds_summary}) for every shot.")
        else:
            report(
                0.335,
                f"No image/video files found in {settings.local_media_dir} "
                "— shots will use generated visuals.",
            )
    else:
        report(0.335, "No local media folder configured — shots will use generated visuals.")

    report(0.34, "Planning beat-synced shots...")
    shots = plan_shots(
        features, mood, transcript=transcript,
        cut_speed_multiplier=content_hints.cut_speed_multiplier,
    )
    report(0.38, f"Planned {len(shots)} shots.")
    check_cancel()

    # --- Stage 5: resolve media per shot (38% - 70%) ------------------------
    resolved_media: dict[int, tuple[str | None, str]] = {}
    local_media_count = 0
    procedural_count = 0
    fetch_span = 0.32  # 38% -> 70%
    resolve_media_for_shot = _make_local_media_resolver(local_media_assets)
    for i, shot in enumerate(shots):
        check_cancel()
        report(
            0.38 + fetch_span * (i / max(1, len(shots))),
            f"Resolving visuals for shot {i + 1}/{len(shots)}...",
        )
        local_path, kind = resolve_media_for_shot()
        if local_path:
            resolved_media[shot.index] = (local_path, kind)
            local_media_count += 1
        else:
            resolved_media[shot.index] = (None, "procedural")
            procedural_count += 1

    if local_media_count:
        report(0.70, f"Visuals resolved: {local_media_count} from your local media folder, {procedural_count} generated.")
    else:
        report(0.70, f"Visuals resolved: {procedural_count} generated (no local media configured).")
    check_cancel()

    # --- Stage 6: render final video (70% - 100%) ---------------------------
    report(0.72, "Assembling final video (encoding)...")
    start = time.time()

    def render_progress(frac: float, msg: str) -> None:
        # video_builder reports its own 0..1 internal progress; remap to 70-100%.
        report(0.70 + 0.30 * frac, msg)

    final_path = assemble_video(
        shots=shots,
        resolved_media=resolved_media,
        audio_path=audio_path,
        mood=mood,
        output_path=output_path,
        target_size=settings.resolution_px,
        fps=settings.fps,
        energy_fn=features.energy_at,
        progress_cb=render_progress,
    )
    render_seconds = time.time() - start
    report(1.0, f"Done! Saved to {final_path}")

    return PipelineResult(
        output_path=final_path,
        mood=mood,
        features=features,
        transcript=transcript,
        shots=shots,
        render_seconds=render_seconds,
        used_local_media_count=local_media_count,
        used_procedural_count=procedural_count,
    )


class PipelineCancelled(Exception):
    """Raised when `cancel_check()` reports the user requested a cancel."""
