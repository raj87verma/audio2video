"""Video assembly: turns planned Shots + resolved media into a finished MP4.

Responsibilities:
  - Ken Burns zoom/pan animation for still images (numpy/PIL, frame-exact).
  - Cover-fit resize/crop for stock video clips and procedural clips.
  - Mood-based color grading (simple numpy color-matrix operations).
  - Crossfade transitions between shots.
  - A synthetic + optional-Freesound SFX layer, beat/highlight-synced,
    mixed underneath the original audio track.
  - Final mux + H.264/AAC MP4 export via MoviePy/FFmpeg.

Every function here is designed to be callable and testable in isolation
(pure numpy/PIL where possible) before being wired into MoviePy clips, so
failures are easy to localize.
"""
from __future__ import annotations

import logging
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    VideoClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.audio.AudioClip import AudioArrayClip

from .mood import MoodProfile
from .procedural_visuals import compose_procedural_frame
from .sfx import synth_sfx_for
from .shot_planner import Shot

log = logging.getLogger(__name__)

CROSSFADE_DURATION = 0.35  # seconds, overlap between consecutive shots
SFX_VOLUME = 0.45          # relative volume of the generated SFX layer vs. original audio
KEN_BURNS_ZOOM = 0.16       # fraction of extra zoom range for the Ken Burns effect

# How many shots are ever "live" in MoviePy/FFmpeg/PIL memory at once while
# assembling. Each shot can hold an open PIL image (still-image Ken Burns
# shots) or, more importantly, an open FFmpeg subprocess pipe (stock-video
# shots, via VideoFileClip). Long audio tracks can plan hundreds of shots
# (e.g. a ~6 minute track easily produces 150-250 shots at typical
# tempos) -- building every single one of those simultaneously and only
# then handing them all to a single concatenate_videoclips() call was
# exhausting memory / OS file-handle limits on real machines (observed:
# an unrecoverable crash partway through building shot clips on a ~200+
# shot track). Rendering in small batches -- each batch fully written to
# a temporary intermediate file and its resources released before the
# next batch starts -- keeps peak resource usage bounded no matter how
# long the source audio is.
SHOT_BATCH_SIZE = 20


# ---------------------------------------------------------------------------
# Color grading
# ---------------------------------------------------------------------------

def apply_color_grade(frame: np.ndarray, grade: str) -> np.ndarray:
    """Apply a simple numpy color-matrix grade to an RGB uint8 frame.

    Deliberately simple (contrast/saturation/tint adjustments) rather than
    true LUTs, so it stays fast enough to run per-frame during rendering.
    """
    f = frame.astype(np.float32)

    if grade == "high_contrast_cool":
        f = (f - 127.5) * 1.25 + 127.5
        f[..., 2] += 15  # push blue
        f[..., 0] -= 8
    elif grade == "warm_vibrant":
        mean = f.mean(axis=-1, keepdims=True)
        f = mean + (f - mean) * 1.3
        f[..., 0] += 14
        f[..., 1] += 4
    elif grade == "moody_desaturated":
        mean = f.mean(axis=-1, keepdims=True)
        f = mean + (f - mean) * 0.55
        f *= 0.88
        f[..., 2] += 8
    elif grade == "soft_pastel":
        f = f * 0.9 + 22
        mean = f.mean(axis=-1, keepdims=True)
        f = mean + (f - mean) * 0.85
    elif grade == "neutral_cinematic":
        f = (f - 127.5) * 1.1 + 127.5
        f[..., 0] += 5
        f[..., 2] += 5
    # unknown grade name -> no-op, still returns a valid frame

    return np.clip(f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Ken Burns (pan/zoom) for still images
# ---------------------------------------------------------------------------

_PAN_ANCHORS = {
    "left": (0.0, 0.5),
    "right": (1.0, 0.5),
    "up": (0.5, 0.0),
    "down": (0.5, 1.0),
    "none": (0.5, 0.5),
}


def make_ken_burns_frame_fn(
    image: Image.Image,
    duration: float,
    target_w: int,
    target_h: int,
    zoom_direction: str = "in",
    pan_direction: str = "none",
    zoom_amount: float = KEN_BURNS_ZOOM,
):
    """Build a `frame_fn(t) -> np.ndarray` for a Ken-Burns-animated still image.

    Precomputes an oversized base image once, then for each `t` crops an
    analytically-interpolated window (size = zoom, position = pan) and
    resizes it to exactly (target_w, target_h). Pure PIL/numpy — no
    MoviePy dependency — so it can be unit tested by calling frame_fn(t)
    directly.
    """
    img = image.convert("RGB")
    iw, ih = img.size
    target_aspect = target_w / target_h

    need_w = target_w * (1 + zoom_amount)
    need_h = target_h * (1 + zoom_amount)
    scale_cover = max(need_w / iw, need_h / ih)
    base_w = max(int(math.ceil(iw * scale_cover)), int(math.ceil(need_w)))
    base_h = max(int(math.ceil(ih * scale_cover)), int(math.ceil(need_h)))
    base_img = img.resize((base_w, base_h), Image.LANCZOS)

    start_w, end_w = (need_w, target_w) if zoom_direction == "in" else (target_w, need_w)
    ax0, ay0 = 0.5, 0.5
    ax1, ay1 = _PAN_ANCHORS.get(pan_direction, (0.5, 0.5))

    def frame_fn(t: float) -> np.ndarray:
        progress = 0.0 if duration <= 0 else float(np.clip(t / duration, 0.0, 1.0))
        w = start_w + (end_w - start_w) * progress
        h = w / target_aspect
        w = min(w, base_w)
        h = min(h, base_h)

        slack_x = max(0.0, base_w - w)
        slack_y = max(0.0, base_h - h)
        ax = ax0 + (ax1 - ax0) * progress
        ay = ay0 + (ay1 - ay0) * progress

        x0 = ax * slack_x
        y0 = ay * slack_y
        box = (x0, y0, x0 + w, y0 + h)
        cropped = base_img.crop(box)
        resized = cropped.resize((target_w, target_h), Image.LANCZOS)
        return np.array(resized)

    return frame_fn


# ---------------------------------------------------------------------------
# Cover-fit resize/crop (for video clips and procedural clips)
# ---------------------------------------------------------------------------

def cover_resize_crop(clip: VideoClip, target_w: int, target_h: int) -> VideoClip:
    """Resize+center-crop a MoviePy clip to exactly (target_w, target_h),
    preserving aspect ratio by cropping the overflow (a "cover" fit, like
    CSS `background-size: cover`).
    """
    cw, ch = clip.w, clip.h
    scale = max(target_w / cw, target_h / ch)
    resized = clip.resize(scale)
    x_center, y_center = resized.w / 2, resized.h / 2
    cropped = resized.crop(
        x_center=x_center, y_center=y_center, width=target_w, height=target_h
    )
    return cropped


# ---------------------------------------------------------------------------
# Per-shot clip construction
# ---------------------------------------------------------------------------

def build_image_shot_clip(
    image_path: str, shot: Shot, target_w: int, target_h: int, fps: int, grade: str
) -> VideoClip:
    img = Image.open(image_path)
    frame_fn = make_ken_burns_frame_fn(
        img, shot.duration, target_w, target_h, shot.zoom_direction, shot.pan_direction
    )

    def make_frame(t):
        return apply_color_grade(frame_fn(t), grade)

    clip = VideoClip(make_frame=make_frame, duration=shot.duration)
    return clip.set_fps(fps)


def build_video_shot_clip(
    video_path: str, shot: Shot, target_w: int, target_h: int, fps: int, grade: str
) -> VideoClip:
    raw = VideoFileClip(video_path, audio=False)
    needed = shot.duration

    if raw.duration >= needed:
        # centered subclip so we don't always start at frame 0 of stock footage
        max_start = max(0.0, raw.duration - needed)
        start = min(max_start, max_start / 2)
        clip = raw.subclip(start, start + needed)
    else:
        # loop short stock clips to cover the shot duration
        loops = int(math.ceil(needed / raw.duration))
        clip = concatenate_videoclips([raw] * loops).subclip(0, needed)

    clip = cover_resize_crop(clip, target_w, target_h)
    clip = clip.fl_image(lambda f: apply_color_grade(f, grade))
    return clip.set_fps(fps).set_duration(needed)


def build_procedural_shot_clip(
    shot: Shot,
    mood: MoodProfile,
    target_w: int,
    target_h: int,
    fps: int,
    energy_fn=None,
    seed: int = 0,
) -> VideoClip:
    """Fallback clip for shots with no downloaded stock media."""

    def make_frame(t):
        abs_t = shot.start + t
        energy = energy_fn(abs_t) if energy_fn else 0.5
        levels = None
        if energy_fn:
            # Sample a short trailing window of the real energy envelope so
            # the waveform-bar overlay visibly reacts to the actual music.
            offsets = np.linspace(-0.9, 0.0, 18)
            levels = np.array([max(0.0, energy_fn(abs_t + off)) for off in offsets])
        frame = compose_procedural_frame(
            target_w, target_h, mood.color_grade, t=abs_t, energy=energy,
            recent_levels=levels, seed=seed + shot.index,
        )
        return np.array(frame)

    clip = VideoClip(make_frame=make_frame, duration=shot.duration)
    return clip.set_fps(fps)


def build_shot_clip(
    shot: Shot,
    local_path: str | None,
    kind: str,
    mood: MoodProfile,
    target_w: int,
    target_h: int,
    fps: int,
    energy_fn=None,
    seed: int = 0,
) -> VideoClip:
    """Dispatch to the right clip builder based on resolved media `kind`.

    `kind` is one of 'image', 'video', or 'procedural' (also used whenever
    `local_path` is falsy/missing, so a bad download never breaks the shot).
    """
    try:
        if kind == "image" and local_path:
            return build_image_shot_clip(local_path, shot, target_w, target_h, fps, mood.color_grade)
        if kind == "video" and local_path:
            return build_video_shot_clip(local_path, shot, target_w, target_h, fps, mood.color_grade)
    except Exception as exc:
        log.warning("Failed to build clip for shot #%d from %s (%s); using procedural fallback", shot.index, local_path, exc)

    return build_procedural_shot_clip(shot, mood, target_w, target_h, fps, energy_fn=energy_fn, seed=seed)


# ---------------------------------------------------------------------------
# SFX timeline mixing
# ---------------------------------------------------------------------------

def _mix_at(track: np.ndarray, sfx: np.ndarray, start_sample: int, gain: float = 1.0) -> None:
    """Additively mix `sfx` into `track` at `start_sample`, safely clipped
    to the track's bounds (handles SFX that start before 0 or extend past
    the end)."""
    n = len(track)
    track_start = max(0, start_sample)
    track_end = min(n, start_sample + len(sfx))
    if track_end <= track_start:
        return
    sfx_start = track_start - start_sample
    sfx_end = sfx_start + (track_end - track_start)
    track[track_start:track_end] += sfx[sfx_start:sfx_end] * gain


def build_sfx_track(shots: list[Shot], total_duration: float, sr: int = 44100) -> np.ndarray:
    """Render the full-length synthetic SFX layer as a mono float32 array.

    - A transition SFX plays at the start of every shot (except a skip on
      the very first shot, to avoid an odd sound before any visuals).
    - A highlight SFX plays at `shot.highlight_time` for shots flagged
      `is_highlight` (strong musical onset in a high-energy shot).
    """
    total_samples = max(1, int(total_duration * sr))
    track = np.zeros(total_samples, dtype=np.float32)

    for i, shot in enumerate(shots):
        if i > 0:
            sfx = synth_sfx_for(shot.energy_level, "transition", sr)
            start_sample = int(shot.start * sr) - int(0.05 * sr)  # slight pre-roll
            _mix_at(track, sfx, start_sample, gain=0.8)

        if shot.is_highlight and shot.highlight_time is not None:
            sfx = synth_sfx_for(shot.energy_level, "highlight", sr)
            start_sample = int(shot.highlight_time * sr)
            _mix_at(track, sfx, start_sample, gain=0.9)

    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 1.0:
        track = track / peak
    return track


# ---------------------------------------------------------------------------
# Full assembly
# ---------------------------------------------------------------------------

def _build_clip_batch(
    shots_batch: list[Shot],
    resolved_media: dict[int, tuple[str | None, str]],
    mood: MoodProfile,
    target_w: int,
    target_h: int,
    fps: int,
    energy_fn,
) -> VideoClip:
    """Build and concatenate (with crossfades) just one batch of shots into
    a single silent video clip. The caller is responsible for closing the
    returned clip once it's done with it (or, in practice here, once it's
    been written out to a temporary file and immediately discarded).
    """
    clips = []
    try:
        for shot in shots_batch:
            local_path, kind = resolved_media.get(shot.index, (None, "procedural"))
            clips.append(
                build_shot_clip(shot, local_path, kind, mood, target_w, target_h, fps, energy_fn=energy_fn)
            )

        if len(clips) > 1:
            faded = [clips[0]] + [c.crossfadein(CROSSFADE_DURATION) for c in clips[1:]]
            batch_video = concatenate_videoclips(faded, method="compose", padding=-CROSSFADE_DURATION)
        else:
            batch_video = clips[0]
        return batch_video
    except Exception:
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
        raise


def assemble_video(
    shots: list[Shot],
    resolved_media: dict[int, tuple[str | None, str]],
    audio_path: str,
    mood: MoodProfile,
    output_path: str,
    target_size: tuple[int, int] = (1920, 1080),
    fps: int = 30,
    energy_fn=None,
    progress_cb=None,
) -> str:
    """Render the final MP4.

    `resolved_media` maps shot.index -> (local_path_or_None, kind) where
    kind is 'image' / 'video' / 'procedural'.
    `energy_fn(t) -> float 0..1` optionally drives procedural-clip
    brightness/particle intensity from the real audio energy envelope.
    `progress_cb(fraction, message)` optionally reports progress (0..1).

    Shots are assembled in bounded-size batches (see `SHOT_BATCH_SIZE`)
    rather than all at once: each batch is fully rendered to a small
    temporary silent-video file and its in-memory MoviePy/PIL/FFmpeg
    resources are released before the next batch starts, and only the
    lightweight temporary *files* are kept around (concatenated at the
    very end). This keeps peak memory/file-handle usage roughly constant
    regardless of how many shots a long track produces, instead of
    scaling with the total shot count.
    """
    target_w, target_h = target_size
    total_shots = len(shots)

    def report(frac, msg):
        log.info("[%.0f%%] %s", frac * 100, msg)
        if progress_cb:
            progress_cb(frac, msg)

    if total_shots == 0:
        raise ValueError("No shots were planned for this audio file; nothing to render.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="audio2video_batches_"))
    batch_paths: list[Path] = []

    try:
        report(0.0, f"Building shot clips (0/{total_shots})...")
        num_batches = math.ceil(total_shots / SHOT_BATCH_SIZE)

        for batch_idx in range(num_batches):
            start_i = batch_idx * SHOT_BATCH_SIZE
            end_i = min(start_i + SHOT_BATCH_SIZE, total_shots)
            shots_batch = shots[start_i:end_i]

            report(
                0.05 + 0.45 * (start_i / total_shots),
                f"Building shots {start_i + 1}-{end_i}/{total_shots}...",
            )

            batch_video = _build_clip_batch(
                shots_batch, resolved_media, mood, target_w, target_h, fps, energy_fn
            )
            try:
                batch_path = tmp_dir / f"batch_{batch_idx:04d}.mp4"
                # No audio yet -- silent intermediate files, muxed with the
                # real audio + SFX only once at the very end. A lower encode
                # preset is fine here since these are throwaway intermediates.
                batch_video.write_videofile(
                    str(batch_path),
                    fps=fps,
                    codec="libx264",
                    audio=False,
                    preset="ultrafast",
                    threads=2,
                    logger=None,
                )
                batch_paths.append(batch_path)
            finally:
                batch_video.close()

            report(
                0.05 + 0.45 * (end_i / total_shots),
                f"Built shots {start_i + 1}-{end_i}/{total_shots}",
            )

        report(0.5, "Joining batches...")
        batch_clips = [VideoFileClip(str(p), audio=False) for p in batch_paths]
        try:
            if len(batch_clips) > 1:
                # Batches were already crossfaded *within* themselves; a
                # straight concatenation (no re-crossfade) between batches
                # avoids double-dipping into the same shot boundary twice.
                video = concatenate_videoclips(batch_clips, method="compose")
            else:
                video = batch_clips[0]

            report(0.6, "Building SFX layer...")
            sfx_track = build_sfx_track(shots, video.duration, sr=44100)
            sfx_stereo = np.stack([sfx_track, sfx_track], axis=1)
            sfx_audio = AudioArrayClip(sfx_stereo, fps=44100).volumex(SFX_VOLUME)

            report(0.7, "Mixing audio...")
            original_audio = AudioFileClip(audio_path)
            try:
                # Match audio length to the (possibly slightly shorter, due
                # to crossfade padding) final video duration so mux doesn't
                # leave a silent/black tail.
                final_duration = min(video.duration, original_audio.duration)
                trimmed_original_audio = original_audio.subclip(0, final_duration)
                trimmed_sfx_audio = sfx_audio.subclip(0, min(final_duration, sfx_audio.duration))
                final_audio = CompositeAudioClip([trimmed_original_audio, trimmed_sfx_audio])

                final_video = video.subclip(0, final_duration).set_audio(final_audio)

                report(0.8, "Encoding final video (this may take a while)...")
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                final_video.write_videofile(
                    output_path,
                    fps=fps,
                    codec="libx264",
                    audio_codec="aac",
                    preset="medium",
                    threads=4,
                    logger=None,
                )
                final_video.close()
            finally:
                original_audio.close()
        finally:
            for c in batch_clips:
                c.close()

        report(1.0, "Done.")
        return output_path

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
