"""Top-level pipeline: audio file in, finished MP4 out.

Wires together every core module in order:

  1. audio_analysis.analyze_audio      — tempo, beats, energy, spectral features
  2. mood.classify_mood                — rule-based mood + keywords + color grade
  3. transcribe.transcribe_audio       — optional local speech-to-text (lyrics)
  4. shot_planner.plan_shots           — beat-aligned shot list
  5. media_fetcher.fetch_best_asset    — per-shot stock photo/video (Pexels/Pixabay)
     (procedural_visuals is used automatically as the video_builder's
     fallback whenever a shot has no resolved asset)
  6. video_builder.assemble_video      — final render + SFX mix + export

Exposes a single `run_pipeline(...)` function with a `progress_cb` hook so
the GUI (or a CLI) can report fine-grained progress without needing to know
about any of the underlying modules.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import OUTPUT_DIR, Settings
from .audio_analysis import AudioFeatures, analyze_audio
from .content_hints import detect_content_hints
from .media_fetcher import download_asset, search_media
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
    used_stock_media_count: int = 0
    used_procedural_count: int = 0


def _noop_progress(fraction: float, message: str) -> None:
    pass


def _default_output_path(audio_path: str) -> str:
    stem = Path(audio_path).stem
    ts = time.strftime("%Y%m%d_%H%M%S")
    return str(OUTPUT_DIR / f"{stem}_{ts}.mp4")


def _make_shot_media_resolver(settings: Settings):
    """Build a per-run `resolve(query, prefer_video) -> (local_path, kind)`
    closure that caches Pexels/Pixabay *search results* per (query,
    prefer_video) pair and round-robins through the downloaded candidates
    on repeat calls for the same query.

    Why this exists: `shot_planner.plan_shots` deliberately cycles through
    a short list of keywords (mood-derived or, since the devotional-theme
    fix, content-derived) across every shot -- a 6-minute track easily
    reuses each of ~6-8 keywords 15-30+ times. Searching *and downloading*
    fresh on every single one of those repeats was measured in practice
    to (a) add one full network round-trip per shot to the render time,
    and (b) trigger Pexels' rate limiting (HTTP 429) partway through a
    long track, since a single search call is issued per repeat instead
    of once per unique keyword.

    The fix here searches once per unique (query, prefer_video) pair,
    downloads up to a handful of the results up front, and then serves
    every repeat of that same query by cycling through the small set of
    already-downloaded local files -- so a long track's many shots for
    the same keyword still get some visual variety (not literally the
    same single clip repeated), while collapsing what used to be N search
    API calls (N = how many shots share that keyword) down to 1.
    """
    search_cache: dict[tuple[str, bool], list] = {}
    cycle_index: dict[tuple[str, bool], int] = {}
    max_downloads_per_query = 4

    def resolve(query: str, prefer_video: bool) -> tuple[str | None, str]:
        cache_key = (query, prefer_video)

        if cache_key not in search_cache:
            candidates = search_media(query, settings, prefer_video=prefer_video)
            downloaded = []
            for asset in candidates:
                if len(downloaded) >= max_downloads_per_query:
                    break
                if download_asset(asset):
                    downloaded.append(asset)
            search_cache[cache_key] = downloaded
            cycle_index[cache_key] = 0

        downloaded = search_cache[cache_key]
        if not downloaded:
            return None, "procedural"

        idx = cycle_index[cache_key] % len(downloaded)
        cycle_index[cache_key] += 1
        asset = downloaded[idx]
        return asset.local_path, asset.kind

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
    shots) instead of continuing. Partial downloads already on disk are left
    in the cache for reuse by a future run.
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
    # Detect a content-level theme (currently: devotional/spiritual) from
    # the filename plus any transcribed lyrics, so shots can be given
    # keywords that actually relate to the song's subject matter instead
    # of only the acoustic mood's generic tempo/loudness-based ones. See
    # content_hints.py's module docstring for why the filename is checked
    # at all (it's the single most reliable free signal for song *topic*).
    content_hints = detect_content_hints(Path(audio_path).name, transcript.full_text)
    if content_hints.is_devotional:
        report(
            0.33,
            f"Devotional/spiritual theme detected ({', '.join(content_hints.matched_terms[:3])}) "
            "— using matching visuals.",
        )

    report(0.34, "Planning beat-synced shots...")
    shots = plan_shots(
        features, mood, transcript=transcript, prefer_video_clips=settings.prefer_video_clips,
        content_keywords=content_hints.keywords or None,
        cut_speed_multiplier=content_hints.cut_speed_multiplier,
    )
    report(0.38, f"Planned {len(shots)} shots.")
    check_cancel()

    # --- Stage 5: fetch stock media per shot (38% - 70%) --------------------
    resolved_media: dict[int, tuple[str | None, str]] = {}
    stock_count = 0
    procedural_count = 0
    fetch_span = 0.32  # 38% -> 70%
    resolve_media_for_shot = _make_shot_media_resolver(settings)
    for i, shot in enumerate(shots):
        check_cancel()
        query = shot.keywords[0] if shot.keywords else mood.keywords[0]
        report(
            0.38 + fetch_span * (i / max(1, len(shots))),
            f"Fetching visuals for shot {i + 1}/{len(shots)} ({query})...",
        )
        local_path, kind = resolve_media_for_shot(query, shot.prefer_video)
        if local_path:
            resolved_media[shot.index] = (local_path, kind)
            stock_count += 1
        else:
            resolved_media[shot.index] = (None, "procedural")
            procedural_count += 1

    report(0.70, f"Visuals resolved: {stock_count} from stock, {procedural_count} generated.")
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
        used_stock_media_count=stock_count,
        used_procedural_count=procedural_count,
    )


class PipelineCancelled(Exception):
    """Raised when `cancel_check()` reports the user requested a cancel."""
