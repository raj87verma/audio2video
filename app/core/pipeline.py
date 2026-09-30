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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import OUTPUT_DIR, Settings
from .audio_analysis import AudioFeatures, analyze_audio
from .content_hints import detect_content_hints
from .credits import write_credits_file
from .media_fetcher import download_asset, search_media
from .mood import MoodProfile, classify_mood
from .shot_planner import Shot, plan_shots
from .transcribe import TranscriptResult, transcribe_audio
from .user_media import scan_user_media
from .video_builder import assemble_video
from .wikimedia_fetcher import WikimediaAsset
from .wikimedia_fetcher import download_asset as download_wikimedia_asset
from .wikimedia_fetcher import search_wikimedia

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
    # Shots that used a file the user placed in their own
    # USER_MEDIA_DIR/<deity>/ folder (see user_media.py) -- these take
    # priority over Wikimedia/Pexels/Pixabay for every shot in a song
    # once any such file is found, so this is typically either 0 (no
    # user media supplied / no deity detected) or equal to the shot count.
    used_user_media_count: int = 0
    # Wikimedia Commons assets actually used in the render, if any --
    # these carry a CC-BY-SA-style attribution requirement that
    # Pexels/Pixabay assets don't, so they're tracked separately for
    # building a credits list (see credits.py).
    wikimedia_assets_used: list[WikimediaAsset] = field(default_factory=list)
    # Path to the auto-written `<video>_credits.txt` attribution file, if
    # `wikimedia_assets_used` was non-empty -- None when no Wikimedia
    # media was used (the common case), matching pre-feature behavior of
    # producing no extra output file.
    credits_file_path: str | None = None


def _noop_progress(fraction: float, message: str) -> None:
    pass


def _default_output_path(audio_path: str) -> str:
    stem = Path(audio_path).stem
    ts = time.strftime("%Y%m%d_%H%M%S")
    return str(OUTPUT_DIR / f"{stem}_{ts}.mp4")


def _make_user_media_resolver(assets: list):
    """Build a `resolve(query, prefer_video) -> (local_path, kind)` closure
    that ignores `query` entirely and just round-robins through `assets`
    (the caller's own supplied photos/videos for the song's detected
    deity -- see `user_media.scan_user_media`).

    Ignoring `query` is deliberate: once a user has supplied their own
    footage of the actual subject, every shot in the song should use it,
    not just the shots whose keyword happens to match a deity term (see
    this module's docstring / the PR description for why the old
    deity-keyword-only routing still left several shots looking
    unrelated). `prefer_video` is honored as a soft preference (video
    assets are tried first if the caller prefers video) but a user who
    only supplied images still gets images, and vice versa.

    Returned `kind` is `"user_image"` / `"user_video"` (not the plain
    `"image"` / `"video"` used for Wikimedia/Pexels/Pixabay) so
    `video_builder.build_shot_clip` can apply the extra treatment meant
    specifically for user-supplied media: a subtle animated VFX overlay
    on stills, and randomized (not fixed-centered) start points on video
    excerpts so a small set of clips still looks varied across many
    shots. See video_builder.py's `build_shot_clip` docstring.
    """
    images = [a for a in assets if a.kind == "image"]
    videos = [a for a in assets if a.kind == "video"]
    index = {"image": 0, "video": 0}

    def resolve(query: str, prefer_video: bool) -> tuple[str | None, str]:
        order = [("video", videos), ("image", images)] if prefer_video else [("image", images), ("video", videos)]
        for kind, pool in order:
            if not pool:
                continue
            i = index[kind] % len(pool)
            index[kind] += 1
            return pool[i].local_path, f"user_{kind}"
        return None, "procedural"

    return resolve


def _make_shot_media_resolver(settings: Settings, deity_queries: frozenset[str] = frozenset()):
    """Build a per-run `resolve(query, prefer_video) -> (local_path, kind)`
    closure that caches search results per (query, prefer_video) pair and
    round-robins through the downloaded candidates on repeat calls for
    the same query.

    Why the caching exists: `shot_planner.plan_shots` deliberately cycles
    through a short list of keywords (mood-derived or, since the
    devotional-theme fix, content-derived) across every shot -- a
    6-minute track easily reuses each of ~6-8 keywords 15-30+ times.
    Searching *and downloading* fresh on every single one of those
    repeats was measured in practice to (a) add one full network
    round-trip per shot to the render time, and (b) trigger Pexels' rate
    limiting (HTTP 429) partway through a long track, since a single
    search call is issued per repeat instead of once per unique keyword.

    The fix here searches once per unique (query, prefer_video) pair,
    downloads up to a handful of the results up front, and then serves
    every repeat of that same query by cycling through the small set of
    already-downloaded local files -- so a long track's many shots for
    the same keyword still get some visual variety (not literally the
    same single clip repeated), while collapsing what used to be N search
    API calls (N = how many shots share that keyword) down to 1.

    `deity_queries`, if given (see `content_hints.ContentHints.
    deity_search_terms`), names the subset of queries that should try
    Wikimedia Commons *first* -- Wikimedia has real, specific coverage of
    named deities that Pexels/Pixabay simply don't (searching "Khatu
    Shyam" on either returns nothing), so for exactly these queries we
    check Wikimedia before falling back to the normal Pexels/Pixabay
    flow. Generic mood/devotional keywords ("hindu temple", "diya lamp",
    ...) are NOT in this set and go straight to Pexels/Pixabay as before
    -- Wikimedia's search quality for broad, generic terms is noisier
    (it's an encyclopedia, not a stock-photo library) and untested at
    that scale, so this integration is deliberately scoped to only the
    specific, verified-good case: named-deity queries.
    """
    search_cache: dict[tuple[str, bool], list] = {}
    cycle_index: dict[tuple[str, bool], int] = {}
    max_downloads_per_query = 4
    # Every distinct Wikimedia asset actually downloaded during this run,
    # in first-use order -- exposed as `resolve.wikimedia_assets_used`
    # after the caller is done, for building the CC-BY-SA attribution/
    # credits list (see credits.py). Pexels/Pixabay assets don't require
    # attribution under their license terms, so only Wikimedia ones are
    # tracked here.
    wikimedia_assets_used: list[WikimediaAsset] = []
    _seen_wikimedia_urls: set[str] = set()

    def _download_wikimedia_candidates(query: str, prefer_video: bool) -> list[WikimediaAsset]:
        downloaded: list[WikimediaAsset] = []
        kinds = ["video", "image"] if prefer_video else ["image", "video"]
        for kind in kinds:
            if len(downloaded) >= max_downloads_per_query:
                break
            for asset in search_wikimedia(query, kind=kind):
                if len(downloaded) >= max_downloads_per_query:
                    break
                if download_wikimedia_asset(asset):
                    downloaded.append(asset)
                    if asset.url not in _seen_wikimedia_urls:
                        _seen_wikimedia_urls.add(asset.url)
                        wikimedia_assets_used.append(asset)
        return downloaded

    def resolve(query: str, prefer_video: bool) -> tuple[str | None, str]:
        cache_key = (query, prefer_video)

        if cache_key not in search_cache:
            downloaded: list = []
            if query in deity_queries and settings.use_wikimedia:
                downloaded = _download_wikimedia_candidates(query, prefer_video)
                if downloaded:
                    log.info(
                        "Wikimedia Commons: found %d asset(s) for deity query %r",
                        len(downloaded), query,
                    )

            if not downloaded:
                candidates = search_media(query, settings, prefer_video=prefer_video)
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

    resolve.wikimedia_assets_used = wikimedia_assets_used
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
    # If the user has placed their own photos/videos of this deity under
    # USER_MEDIA_DIR (see user_media.py), those take absolute priority --
    # every shot in the song uses them, bypassing Wikimedia/Pexels/
    # Pixabay/procedural entirely, since the user has explicitly supplied
    # footage of the actual subject. Checked before deciding whether to
    # even mention Wikimedia in the progress log, so the messaging is
    # accurate about which source is actually being used.
    user_media_assets = scan_user_media(content_hints.deity) if content_hints.deity else []

    if user_media_assets:
        kinds_summary = f"{sum(1 for a in user_media_assets if a.kind == 'video')} video(s), {sum(1 for a in user_media_assets if a.kind == 'image')} photo(s)"
        report(
            0.335,
            f"Using your own {content_hints.deity} media ({kinds_summary}) for every shot.",
        )
    elif content_hints.deity:
        report(0.335, f"Specific deity detected: {content_hints.deity} — checking Wikimedia Commons for real footage.")

    # When a specific deity was identified, put its Wikimedia-tuned search
    # terms *ahead of* the generic devotional keywords in the cycle shots
    # draw from -- this biases most shots toward the actual named subject
    # (e.g. "Khatu Shyam") while still mixing in generic devotional
    # visuals (temple/diya/prayer) for variety across a long track. This
    # keyword list is still passed to plan_shots() even when user_media_assets
    # is non-empty (it drives the progress-log "fetching visuals for X"
    # message and stays available if the user later empties the folder),
    # but the resolver built below ignores it entirely in that case.
    combined_keywords = (content_hints.deity_search_terms + content_hints.keywords) or None

    report(0.34, "Planning beat-synced shots...")
    shots = plan_shots(
        features, mood, transcript=transcript, prefer_video_clips=settings.prefer_video_clips,
        content_keywords=combined_keywords,
        cut_speed_multiplier=content_hints.cut_speed_multiplier,
        # Devotional lyrics ("tera", "karo", "jai", "kalyan", ...) are
        # invocation/grammar words, not visual descriptions -- if a
        # devotional song has clear vocals, transcribed lyric keywords
        # would otherwise silently override the correctly-detected
        # devotional/deity keywords for almost every shot (see
        # shot_planner.plan_shots' docstring). Only devotional content
        # flips this priority; ordinary songs still prefer their own
        # (often genuinely descriptive) lyric keywords as before.
        prioritize_content_keywords=content_hints.is_devotional,
    )
    report(0.38, f"Planned {len(shots)} shots.")
    check_cancel()

    # --- Stage 5: fetch stock media per shot (38% - 70%) --------------------
    resolved_media: dict[int, tuple[str | None, str]] = {}
    stock_count = 0
    procedural_count = 0
    user_media_count = 0
    fetch_span = 0.32  # 38% -> 70%
    if user_media_assets:
        resolve_media_for_shot = _make_user_media_resolver(user_media_assets)
    else:
        resolve_media_for_shot = _make_shot_media_resolver(
            settings, deity_queries=frozenset(content_hints.deity_search_terms)
        )
    for i, shot in enumerate(shots):
        check_cancel()
        query = shot.keywords[0] if shot.keywords else mood.keywords[0]
        report(
            0.38 + fetch_span * (i / max(1, len(shots))),
            f"Fetching visuals for shot {i + 1}/{len(shots)} ({query})...",
        )
        local_path, kind = resolve_media_for_shot(query, shot.prefer_video)
        if local_path and kind in ("user_image", "user_video"):
            resolved_media[shot.index] = (local_path, kind)
            user_media_count += 1
        elif local_path:
            resolved_media[shot.index] = (local_path, kind)
            stock_count += 1
        else:
            resolved_media[shot.index] = (None, "procedural")
            procedural_count += 1

    if user_media_count:
        report(0.70, f"Visuals resolved: {user_media_count} from your own media.")
    else:
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

    # If any Wikimedia Commons media was used, write its CC-license
    # attribution alongside the video -- see credits.py's module
    # docstring for why this is required (Wikimedia media, unlike
    # Pexels/Pixabay, is almost always attribution-licensed). A no-op
    # (returns None, writes nothing) when the list is empty, which is the
    # common case for renders that never matched a specific named deity.
    # The user-media resolver (used when the user supplied their own
    # files) has no such attribute at all -- the user's own media needs
    # no Wikimedia-style attribution tracking, so this is simply skipped
    # in that case.
    wikimedia_assets_used = getattr(resolve_media_for_shot, "wikimedia_assets_used", [])
    credits_file_path = write_credits_file(wikimedia_assets_used, final_path)
    if credits_file_path:
        report(1.0, f"Done! Saved to {final_path} (see {Path(credits_file_path).name} for media credits)")
    else:
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
        used_user_media_count=user_media_count,
        wikimedia_assets_used=wikimedia_assets_used,
        credits_file_path=credits_file_path,
    )


class PipelineCancelled(Exception):
    """Raised when `cancel_check()` reports the user requested a cancel."""
