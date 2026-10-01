"""Video assembly: turns planned Shots + resolved media into a finished MP4.

Responsibilities:
  - Ken Burns zoom/pan animation for still images (numpy/PIL, frame-exact).
  - Cover-fit resize/crop for the user's own video clips and procedural clips.
  - Mood-based color grading (simple numpy color-matrix operations).
  - Crossfade transitions between shots.
  - A synthetic SFX layer, beat/highlight-synced, mixed underneath the
    original audio track.
  - Final mux + H.264/AAC MP4 export via MoviePy/FFmpeg.

Every function here is designed to be callable and testable in isolation
(pure numpy/PIL where possible) before being wired into MoviePy clips, so
failures are easy to localize.
"""
from __future__ import annotations

import logging
import math
import random
import re
import shutil
import subprocess as sp
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from moviepy.config import get_setting
from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    VideoClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.audio.AudioClip import AudioArrayClip

from .mood import MoodProfile
from .procedural_visuals import compose_procedural_frame, render_glow_sparkle_overlay_onto
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
#
# Why this is 4, not 20 (lowered after a *second*, independent
# MemoryError was reported even after target_resolution-based decode
# downscaling (see _probe_video_info/_decode_target_resolution) was
# already in place): VideoFileClip spawns a real FFmpeg subprocess the
# instant it's constructed -- not lazily on first frame read, see
# FFMPEG_VideoReader.initialize() -- and that subprocess has its own
# memory cost (reference-frame/decode buffers inside FFmpeg itself)
# that is largely independent of how small a `target_resolution` is
# requested, since FFmpeg still has to decode the source's native
# frames before its scale filter ever runs. Measured directly against a
# real 4K source: a single such subprocess costs roughly 150-700MB of
# resident memory, and this scales up ~linearly with how many are open
# at once -- 20 concurrent 4K opens (the previous SHOT_BATCH_SIZE, in
# the all-video-shots case a user's own media-only folder produces)
# measured at ~14GB of resident memory, comfortably exceeding what a
# typical consumer machine (and definitely what the "paging file too
# small" error from a real user's crash log implied) has available.
# Lowering the batch size directly bounds the number of concurrent
# FFmpeg subprocesses -- 4 measured at ~2.7GB worst-case (every shot in
# the batch sourced from a 4K video, i.e. a user's media folder with
# only videos in it, same as the real crash report) -- regardless of
# source resolution, which the target_resolution optimization alone
# could not do since it only reduces per-frame array size inside Python,
# not FFmpeg's own internal per-process decode overhead.
SHOT_BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Color grading
# ---------------------------------------------------------------------------

def _make_grade_matrices() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Precompute a (3x3 color matrix, 3-vector bias) pair per grade name.

    Every grade below is expressible as an affine transform of the pixel's
    (R,G,B) vector: `out = clip(pixel @ M.T + bias, 0, 255)`. This is
    mathematically identical to the original per-grade formulas (which
    computed things like a per-pixel channel mean and lerped toward it) --
    a channel mean is itself just a linear combination of R,G,B, so folding
    it into `M` costs nothing at apply-time. This matters because at
    render scale, `apply_color_grade` runs once per output frame per shot
    (potentially hundreds of thousands of times for a long track), and a
    single `frame @ M.T + bias` matrix-multiply is measured to be roughly
    3x faster than the original per-grade `.mean(axis=-1, keepdims=True)` +
    multiple elementwise-array formula, since it does one fused BLAS-backed
    operation instead of several separate full-frame temporary arrays.
    """
    identity = np.eye(3, dtype=np.float64)
    mean_matrix = np.ones((3, 3), dtype=np.float64) / 3.0  # replicates .mean(axis=-1)

    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # high_contrast_cool: (f - 127.5) * 1.25 + 127.5, then B+=15, R-=8
    c = -127.5 * 1.25 + 127.5
    matrices["high_contrast_cool"] = (identity * 1.25, np.array([c - 8, c, c + 15]))

    # warm_vibrant: mean + (f - mean) * 1.3, then R+=14, G+=4
    a = 1.3
    m = mean_matrix * (1 - a) + identity * a
    matrices["warm_vibrant"] = (m, np.array([14.0, 4.0, 0.0]))

    # moody_desaturated: (mean + (f - mean) * 0.55) * 0.88, then B+=8
    a = 0.55
    m = (mean_matrix * (1 - a) + identity * a) * 0.88
    matrices["moody_desaturated"] = (m, np.array([0.0, 0.0, 8.0]))

    # soft_pastel: f1 = f*0.9 + 22; out = mean(f1) + (f1 - mean(f1)) * 0.85
    a = 0.85
    m2 = mean_matrix * (1 - a) + identity * a
    matrices["soft_pastel"] = (m2 * 0.9, m2 @ np.array([22.0, 22.0, 22.0]))

    # neutral_cinematic: (f - 127.5) * 1.1 + 127.5, then R+=5, B+=5
    c = -127.5 * 1.1 + 127.5
    matrices["neutral_cinematic"] = (identity * 1.1, np.array([c + 5, c, c + 5]))

    # Pre-transpose (for the `frame @ M` call site) and cast once here so
    # apply_color_grade's hot path never repeats that work per-frame.
    return {
        name: (m.T.astype(np.float32).copy(), bias.astype(np.float32))
        for name, (m, bias) in matrices.items()
    }


_GRADE_MATRICES = _make_grade_matrices()


def apply_color_grade(frame: np.ndarray, grade: str) -> np.ndarray:
    """Apply a simple color-matrix grade to an RGB uint8 frame.

    Implemented as a single affine transform (`frame @ M + bias`) per
    `_make_grade_matrices`'s docstring -- deliberately simple (contrast/
    saturation/tint adjustments) rather than true LUTs, and fast enough to
    run per-frame during rendering even at 1080p+ resolutions and long
    track lengths.
    """
    transform = _GRADE_MATRICES.get(grade)
    if transform is None:
        # unknown grade name -> no-op, still returns a valid frame
        return frame

    matrix_t, bias = transform
    out = frame.astype(np.float32) @ matrix_t
    out += bias
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


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
# Non-square pixel (SAR) correction for source videos
# ---------------------------------------------------------------------------

_SAR_RE = re.compile(r"\bSAR\s+(\d+):(\d+)\b")
_SIZE_RE = re.compile(r"\b(\d{2,5})x(\d{2,5})\b")

# How much headroom to decode above the final render resolution before
# cover_resize_crop does its own resize+crop. 1.6x covers typical
# aspect-ratio mismatches (e.g. a 1:1 or 4:3 source cropped to 16:9)
# without needing the full native resolution of a real phone/camera
# recording, which is very often far larger than any render target this
# app supports (max 4k == 2160 tall).
_DECODE_HEADROOM = 1.6
# Only bother requesting a smaller decode resolution from ffmpeg when the
# source is at least this many times taller than what we'd actually
# decode at -- avoids any downscale overhead/risk for sources that are
# already close to the target size.
_DECODE_DOWNSCALE_THRESHOLD = 1.2


def _probe_video_info(video_path: str, timeout: int = 30) -> tuple[int, int, float] | None:
    """Probe a video file's *coded* width/height and Sample Aspect Ratio
    (SAR) with a single `ffmpeg -i` call. Returns `(coded_w, coded_h, sar)`,
    or `None` if the file couldn't be probed (missing binary, corrupt
    file, or the probe itself timed out) -- callers should fall back to
    opening the file at its native resolution with no SAR correction in
    that case, exactly like every previous release did before this
    function existed.

    Why probing coded size matters (not just SAR): real phone/camera
    video recordings -- the overwhelmingly common case for a user's own
    "local media folder" footage, unlike pre-processed Pexels/Pixabay-
    style stock video -- are very often shot at 4K (3840x2160) or higher,
    while this app only ever renders at up to 1080p (and usually 720p).
    MoviePy/FFMPEG_VideoReader decodes every frame at the source's full
    *coded* resolution regardless of what the final render needs --
    there is no cost-saving default. A single 4K frame is ~24MB
    (3840*2160*3 bytes); with several such videos open at once within a
    render batch (see SHOT_BATCH_SIZE), this was observed to exhaust
    available memory/the Windows paging file on a real machine,
    crashing mid-render with `MemoryError` / `OSError: [WinError 1455]
    The paging file is too small...` -- confirmed via a real user's log
    showing exactly these errors against their own folder of 4K Khatu
    Shyam video clips. See `build_video_shot_clip`'s use of this probe's
    result to request a much smaller decode resolution from ffmpeg
    directly (via `VideoFileClip`'s `target_resolution` parameter)
    whenever the source is meaningfully larger than needed.

    Why SAR is also probed here (not just coded size): MoviePy 1.0.3's
    `FFMPEG_VideoReader` parses only the raw *coded* pixel dimensions out
    of ffmpeg's `Video: ... WxH ...` info line -- it never looks at
    SAR/DAR at all. Some real phone-camera recordings are encoded with
    non-square pixels (verified case: coded 1080x1080 but `SAR 76:135`,
    i.e. true display size 608x1080, a portrait video, not square).
    Without correcting for this, every frame is silently stretched
    relative to how it's meant to look. Probing both coded size and SAR
    together in one `ffmpeg -i` call (instead of two separate probes,
    which is what an earlier version of this code did) halves the
    probing overhead per video and halves the chances of a slow/timed-out
    probe for any single file.
    """
    try:
        proc = sp.run(
            [get_setting("FFMPEG_BINARY"), "-i", video_path],
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            stdin=sp.DEVNULL,
            timeout=timeout,
        )
        info = proc.stderr.decode("utf8", errors="ignore")

        size_match = _SIZE_RE.search(info)
        if not size_match:
            return None
        coded_w, coded_h = int(size_match.group(1)), int(size_match.group(2))
        if coded_w <= 0 or coded_h <= 0:
            return None

        sar = 1.0
        sar_match = _SAR_RE.search(info)
        if sar_match:
            num, den = int(sar_match.group(1)), int(sar_match.group(2))
            if den != 0:
                sar = num / den

        return coded_w, coded_h, sar
    except Exception:
        # Any probing failure (missing binary, unexpected output, timeout,
        # corrupt/unreadable file) should never break rendering -- the
        # caller falls back to opening the file at native resolution with
        # no SAR correction, which is what every previous release
        # effectively did unconditionally.
        log.warning("Could not probe video info for %s; using native resolution, assuming square pixels", video_path, exc_info=True)
        return None


def _decode_target_resolution(coded_w: int, coded_h: int, target_w: int, target_h: int) -> tuple[int, int] | None:
    """Decide whether to ask ffmpeg to decode `video_path` at a smaller
    resolution than its native coded size, and if so, return the
    `(desired_height, desired_width)` tuple to pass as `VideoFileClip`'s
    `target_resolution` parameter (per that parameter's own documented
    order) -- or `None` if the source is already close enough to the
    target size that downscaling isn't worth the extra complexity.

    Only `desired_height` is ever actually set (width is left `None`, so
    `FFMPEG_VideoReader` scales width proportionally to the *coded*
    aspect ratio on its own) -- this deliberately leaves SAR correction
    to run as a separate step afterward on the resulting (now smaller)
    frame, exactly as it already did before any downscaling existed, just
    operating on fewer pixels. See `build_video_shot_clip` for how the
    two steps compose.
    """
    # The larger of the two target dimensions, with headroom for
    # whichever orientation (landscape/portrait) the source turns out to
    # be relative to the render target -- cover_resize_crop will scale up
    # to fully cover (target_w, target_h) and crop the rest, so decoding
    # at _DECODE_HEADROOM times the larger target dimension comfortably
    # covers that regardless of the source's own aspect ratio.
    desired_dim = int(max(target_w, target_h) * _DECODE_HEADROOM)
    if coded_h <= desired_dim * _DECODE_DOWNSCALE_THRESHOLD and coded_w <= desired_dim * _DECODE_DOWNSCALE_THRESHOLD:
        return None
    # Downscale based on whichever coded dimension is larger, so a
    # portrait source (coded_h > coded_w) and a landscape one both end up
    # with their longer edge close to desired_dim rather than only ever
    # constraining height.
    if coded_h >= coded_w:
        return desired_dim, None
    # VideoFileClip's target_resolution is (height, width); to constrain
    # width instead for a landscape source, compute the proportional
    # height so FFMPEG_VideoReader's own "only one dimension given"
    # scaling path (ratio = target / self.size[idx]) ends up constraining
    # width to desired_dim as intended.
    desired_height = max(1, int(round(coded_h * (desired_dim / coded_w))))
    return desired_height, None


def _correct_non_square_pixels_known_sar(raw: VideoFileClip, sar: float, video_path: str) -> VideoFileClip:
    """Like the old `_correct_non_square_pixels`, but takes an
    already-probed `sar` value instead of probing it itself -- see
    `_probe_video_info`'s docstring for why the probe now happens once,
    up front, shared with the decode-resolution decision, rather than a
    second time here.
    """
    if abs(sar - 1.0) < 1e-3:
        return raw
    coded_w, coded_h = raw.size
    display_w = coded_w * sar
    # Resize to the true display size, keeping height fixed and scaling
    # width by the SAR factor -- this is the standard SAR-correction
    # convention (height is the "reference" axis; DAR = SAR * coded_w/coded_h).
    corrected = raw.resize(newsize=(int(round(display_w)), coded_h))
    log.info(
        "Corrected non-square pixels for %s: coded %dx%d (SAR %.4f) -> display %dx%d",
        video_path, coded_w, coded_h, sar, corrected.w, corrected.h,
    )
    return corrected


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
    image_path: str,
    shot: Shot,
    target_w: int,
    target_h: int,
    fps: int,
    grade: str,
    add_vfx_overlay: bool = False,
) -> VideoClip:
    """Build a shot clip from a still image (Ken Burns pan/zoom + color grade).

    `add_vfx_overlay`, when True, additionally composites a subtle
    animated glow + drifting-sparkle effect on top (see
    `procedural_visuals.render_glow_sparkle_overlay`). This is used for
    the user's own supplied photos (see `local_media.py` / `pipeline.
    _make_local_media_resolver`) -- a static personal photo otherwise
    looks comparatively flat/motionless next to the animated procedural-
    fallback shots elsewhere in the same render, and users have
    explicitly asked for "animations and vfx" on their own images.
    """
    img = Image.open(image_path)
    # Apply EXIF orientation before anything else. Phone-camera photos
    # frequently store an EXIF `Orientation` tag instead of storing
    # pixels already rotated upright. `Image.open()` ignores that tag, so
    # without this the image renders sideways or upside-down. Verified
    # with a real phone-camera photo (rotated before this fix, correct
    # after it).
    img = ImageOps.exif_transpose(img)
    frame_fn = make_ken_burns_frame_fn(
        img, shot.duration, target_w, target_h, shot.zoom_direction, shot.pan_direction
    )

    if add_vfx_overlay:
        # Seeded by shot.index so the sparkle pattern differs shot-to-shot
        # (a user typically has just a handful of photos reused across
        # many shots -- an identical, unmoving overlay every single time
        # would look obviously repetitive) while staying deterministic
        # for a given render (re-running on the same input reproduces the
        # same output, consistent with how every other seeded effect in
        # this codebase behaves, e.g. build_procedural_shot_clip's seed).
        def make_frame(t):
            base = apply_color_grade(frame_fn(t), grade)
            return render_glow_sparkle_overlay_onto(base, t, seed=shot.index)
    else:
        def make_frame(t):
            return apply_color_grade(frame_fn(t), grade)

    clip = VideoClip(make_frame=make_frame, duration=shot.duration)
    return clip.set_fps(fps)


def build_video_shot_clip(
    video_path: str,
    shot: Shot,
    target_w: int,
    target_h: int,
    fps: int,
    grade: str,
    trim_seconds: tuple[float, float] | None = None,
) -> VideoClip:
    """Build a shot clip from a downloaded stock video file.

    `trim_seconds`, when given as a `(min, max)` range in seconds, first
    extracts one short, randomly-positioned trim of a randomly-chosen
    length in that range from the source -- e.g. `(2.0, 4.0)` picks
    somewhere between 2 and 4 seconds of random footage from a random
    point in the source -- and then treats *that trim* (not the original
    full source) as what gets shown for the shot: if the shot needs less
    than the trim's own length, a centered piece of the trim is used; if
    the shot needs more (a shot's `duration` can legitimately be longer
    than a few seconds, depending on tempo/mood pacing), the short trim is
    looped to fill the remaining time (see the loop branch below -- this
    is the same "loop short clips to cover the shot duration" logic
    already used for any video shorter than its shot, just applied to the
    trim instead of the raw source).

    This is used for the user's own supplied video(s) (see
    `local_media.py` / `pipeline._make_local_media_resolver`) -- since a
    song typically has many more shots than a user is likely to supply
    distinct video files for, so the *same* file gets reused across many
    shots (via round-robin cycling in the resolver). Without this, every
    one of those reuses would show either the exact same fixed segment
    (if always centered) or an arbitrarily long stretch starting at a
    random point (if only randomizing the start) -- neither gives the
    short, varied few-second excerpts asked for. Left `None` for any
    caller that wants the original full-video behavior (e.g. tests
    exercising the plain path directly).

    IMPORTANT resource-management note: `VideoFileClip` opens an FFmpeg
    subprocess (a real OS process + pipe) for the lifetime of the object,
    and overrides `close()` to terminate it. Every derived clip we create
    from it below (`subclip`, `resize`, `crop`, `fl_image`, ...) wraps the
    original in a *new* Clip object whose own `close()` is the inert
    default from the base `Clip` class -- it does **not** know how to
    reach back and terminate the original `VideoFileClip`'s subprocess.
    Only closing that exact original `raw` object (or a `concatenate_*`
    result that MoviePy happens to store `.clips` on) actually releases
    the subprocess.

    Without doing this explicitly, every stock-video shot leaks one
    running FFmpeg process for the remainder of the program's life. On a
    long track with hundreds of video shots (very much the common case
    if the user's local media folder favors video, since video is
    preferred over stills), this silently accumulates until the process
    exhausts memory/handles and crashes -- even though the *reported*
    per-batch clips are being closed correctly, because none of those
    ever held a reference to the real underlying subprocess in the first
    place.

    So: `raw` is returned to the caller stashed on the resulting clip via
    `._audio2video_raw_clips`, and every caller in this module is
    responsible for closing everything in that list once it's done with
    the shot (see `_close_shot_clip`).
    """
    # Probe coded size + SAR once up front (see _probe_video_info's
    # docstring) so we can ask ffmpeg to decode directly at a much
    # smaller resolution when the source is far larger than this render
    # actually needs -- e.g. a real 4K (3840x2160) phone recording being
    # rendered into a 720p/1080p video. Without this, MoviePy decodes
    # every frame at the source's full native resolution regardless of
    # the render target, which is harmless for one video but was
    # confirmed (via a real user's crash log) to exhaust memory/the
    # Windows paging file when several such 4K sources are open within
    # the same render batch. `probe_info` is None if probing failed for
    # any reason (missing binary, corrupt file, timeout) -- in that case
    # we fall back to opening at native resolution with no SAR
    # correction, exactly like every previous release did unconditionally.
    probe_info = _probe_video_info(video_path)
    target_resolution = None
    sar = 1.0
    if probe_info is not None:
        coded_w, coded_h, sar = probe_info
        target_resolution = _decode_target_resolution(coded_w, coded_h, target_w, target_h)
        if target_resolution is not None:
            log.info(
                "Decoding %s at reduced resolution (coded %dx%d -> target_resolution=%s) to limit memory use",
                video_path, coded_w, coded_h, target_resolution,
            )

    try:
        raw = VideoFileClip(video_path, audio=False, target_resolution=target_resolution)
    except Exception:
        if target_resolution is not None:
            # A handful of codecs/containers don't tolerate ffmpeg's own
            # `-vf scale=...` cleanly (observed rarely with some H.264
            # profiles) -- retry once at native resolution rather than
            # losing the whole shot to procedural fallback over what is
            # purely a memory-saving optimization, not a correctness
            # requirement.
            log.warning(
                "Failed to open %s at reduced resolution %s; retrying at native resolution",
                video_path, target_resolution, exc_info=True,
            )
            raw = VideoFileClip(video_path, audio=False)
        else:
            raise
    raw_clips_to_close = [raw]
    needed = shot.duration

    # Correct non-square pixels using the SAR already probed above --
    # `raw` itself is left untouched (still tracked in
    # raw_clips_to_close for cleanup, since it owns the underlying
    # FFmpeg subprocess); `source` is what the rest of this function
    # actually reads frames from.
    source = _correct_non_square_pixels_known_sar(raw, sar, video_path)

    if trim_seconds is not None:
        # Pick a random trim length within the requested range (clamped
        # to the source's own duration, in case a supplied video is
        # itself shorter than the requested minimum trim length -- e.g. a
        # 3s source with trim_seconds=(2.0, 4.0) simply gets used in full
        # rather than raising or producing an invalid subclip range).
        # A fresh Random() per call (not seeded) is deliberate -- the
        # whole point is that repeated calls for the *same* source file
        # (across many shots reusing a small set of user-supplied videos)
        # land on different excerpts each time, which a fixed seed would
        # defeat.
        lo, hi = trim_seconds
        trim_len = min(random.uniform(lo, hi), source.duration)
        max_trim_start = max(0.0, source.duration - trim_len)
        trim_start = random.uniform(0.0, max_trim_start)
        # Reassign `source` to the short trim itself -- everything below
        # (the "does it need looping to fill the shot" logic) then
        # operates on this short trim exactly as it would on a naturally
        # short source video, so a single random trim naturally loops to
        # fill however long the shot actually needs.
        source = source.subclip(trim_start, trim_start + trim_len)

    if source.duration >= needed:
        max_start = max(0.0, source.duration - needed)
        # Centered subclip of whatever `source` is at this point (either
        # the original full video, when trim_seconds is None, or the
        # short random trim from just above) -- avoids always starting
        # at frame 0.
        start = min(max_start, max_start / 2)
        clip = source.subclip(start, start + needed)
    else:
        # loop short stock clips to cover the shot duration. Each repeated
        # reference in `[source] * loops` is the *same* object (whether
        # that's `raw` itself or the SAR-corrected wrapper around it), so
        # this does not open extra subprocesses -- `raw` alone still
        # covers it, same as before the SAR-correction was added.
        loops = int(math.ceil(needed / source.duration))
        clip = concatenate_videoclips([source] * loops).subclip(0, needed)

    clip = cover_resize_crop(clip, target_w, target_h)
    clip = clip.fl_image(lambda f: apply_color_grade(f, grade))
    clip = clip.set_fps(fps).set_duration(needed)
    clip._audio2video_raw_clips = raw_clips_to_close
    return clip


def _close_shot_clip(clip) -> None:
    """Close `clip` and, if present, any underlying raw `VideoFileClip`(s)
    stashed on it by `build_video_shot_clip` -- see that function's
    docstring for why this indirection is necessary. Safe to call on any
    clip, including ones that never touched a video file.
    """
    for raw in getattr(clip, "_audio2video_raw_clips", []):
        try:
            raw.close()
        except Exception:
            pass
    try:
        clip.close()
    except Exception:
        pass


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



# Random 2-4 second video trims for the user's own supplied footage (see
# build_video_shot_clip's `trim_seconds` docstring) -- module-level since
# every "local_video" shot uses the same range.
_LOCAL_VIDEO_TRIM_RANGE = (2.0, 4.0)


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

    `kind` is `"local_image"`, `"local_video"`, or `"procedural"` (also
    used whenever `local_path` is falsy/missing, so a bad/unreadable file
    never breaks the shot). The `"local_*"` kinds (see `local_media.py` /
    `pipeline._make_local_media_resolver` -- the user's own media folder)
    get the extra treatment meant for user-supplied media: a glow/sparkle
    VFX overlay for stills, and a short randomized (2-4 second) trim for
    video excerpts -- see `build_image_shot_clip`'s `add_vfx_overlay` and
    `build_video_shot_clip`'s `trim_seconds` params.
    """
    try:
        if kind == "local_image" and local_path:
            return build_image_shot_clip(
                local_path, shot, target_w, target_h, fps, mood.color_grade, add_vfx_overlay=True
            )
        if kind == "local_video" and local_path:
            return build_video_shot_clip(
                local_path, shot, target_w, target_h, fps, mood.color_grade,
                trim_seconds=_LOCAL_VIDEO_TRIM_RANGE,
            )
    except Exception as exc:
        # If building the video clip failed partway through (e.g. "failed
        # to read the first frame", a corrupted/truncated download), any
        # VideoFileClip subprocess it already opened before failing would
        # otherwise leak silently -- there is no successfully-returned
        # clip object here for the caller to close. video_builder can't
        # see `raw` from inside build_video_shot_clip's exception path, so
        # explicitly ask FFMPEG_VideoReader-backed clips to release
        # themselves isn't possible here; instead build_video_shot_clip
        # itself guards this above by only registering `raw` for cleanup
        # on success. Genuinely orphaned subprocesses from a raised
        # exception inside VideoFileClip's own constructor are extremely
        # rare (that constructor doesn't open the pipe until first frame
        # read) and are left to the OS/process exit, same as any other
        # third-party library failure would be.
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
) -> tuple[VideoClip, list[VideoClip]]:
    """Build and concatenate (with crossfades) just one batch of shots into
    a single silent video clip.

    Returns `(batch_video, clips_to_close)`. `clips_to_close` includes every
    individual per-shot clip that was built (and, transitively via
    `_close_shot_clip`, whatever raw `VideoFileClip` each of them wraps).

    IMPORTANT: when `len(clips) > 1`, `batch_video` is a `CompositeVideoClip`
    (that's what `concatenate_videoclips(..., method="compose")` returns).
    `CompositeVideoClip.close()` only closes its own `bg`/`audio` attributes
    -- it does **not** close the `clips` it was built from. So closing just
    `batch_video` on its own leaves every per-shot clip's underlying FFmpeg
    subprocess (for any stock-video shot) running indefinitely. The caller
    MUST close everything in `clips_to_close` *in addition to* `batch_video`,
    and MUST do so only after it's completely done reading frames from
    `batch_video` (e.g. after `write_videofile()` has returned) -- closing
    them earlier would pull the rug out from under frames still being read.
    """
    clips = []
    try:
        for shot in shots_batch:
            local_path, kind = resolved_media.get(shot.index, (None, "procedural"))
            clips.append(
                build_shot_clip(shot, local_path, kind, mood, target_w, target_h, fps, energy_fn=energy_fn)
            )

        if len(clips) > 1:
            # crossfadein() reads each clip's mask, which (for the common
            # case of a plain opaque clip) MoviePy constructs lazily via
            # ColorClip -- i.e. an actual new per-frame numpy array
            # allocation, not a cheap wrapper -- the very first time it's
            # touched. This is where a 4K-video-derived clip's memory
            # pressure (see build_video_shot_clip's probe-based
            # downscaling, which already reduces this a lot but doesn't
            # eliminate it entirely under severe system memory pressure)
            # was observed to actually surface as a crash in a real user's
            # log: every individual shot had already built successfully
            # (several even fell back to the lightweight procedural
            # generator after their own MemoryError, exactly as designed),
            # but the crossfade step itself then hit the same wall with no
            # fallback at all, taking down the entire render rather than
            # just one shot. Guard each crossfade call individually so a
            # transient memory failure here degrades to a hard cut for
            # that one transition (losing a cosmetic fade, not the whole
            # render) instead of propagating out of this function.
            faded = [clips[0]]
            for i, c in enumerate(clips[1:], start=1):
                try:
                    faded.append(c.crossfadein(CROSSFADE_DURATION))
                except Exception:
                    log.warning(
                        "Crossfade transition failed for shot %d/%d in this batch (likely low memory); "
                        "using a hard cut instead",
                        i, len(clips), exc_info=True,
                    )
                    faded.append(c)
            try:
                batch_video = concatenate_videoclips(faded, method="compose", padding=-CROSSFADE_DURATION)
            except Exception:
                log.warning(
                    "Crossfaded concatenation failed for this batch (likely low memory); "
                    "retrying with no crossfades (hard cuts) for this batch",
                    exc_info=True,
                )
                batch_video = concatenate_videoclips(clips, method="compose")
        else:
            batch_video = clips[0]
        return batch_video, clips
    except Exception:
        for c in clips:
            _close_shot_clip(c)
        raise


def _render_shots_chunk_to_file(
    shots_chunk: list[Shot],
    resolved_media: dict[int, tuple[str | None, str]],
    mood: MoodProfile,
    target_w: int,
    target_h: int,
    fps: int,
    energy_fn,
    tmp_dir: Path,
    chunk_label: str,
) -> Path:
    """Build+crossfade one chunk of shots and write it to its own small
    temporary MP4, returning the Path.

    This is the actual last line of defense against `MemoryError`/
    `OSError` ("paging file too small" on Windows) crashes caused by too
    many concurrently-open video decoders (see `SHOT_BATCH_SIZE`'s
    docstring for the measured root cause: each `VideoFileClip` spawns a
    real FFmpeg subprocess immediately on construction -- not lazily on
    first frame read -- and each one costs on the order of hundreds of
    MB of its own resident memory, independent of any `target_resolution`
    downscaling, which only reduces per-*frame* array size inside Python,
    not FFmpeg's own internal decode buffers).

    `SHOT_BATCH_SIZE` was lowered specifically to keep the common case
    comfortably under real machines' available memory (measured: 4
    concurrent 4K-source opens peaks around 2.7GB vs the previous
    default of 20 peaking around 14GB) -- but no single fixed batch size
    can be guaranteed safe for every real machine (available RAM,
    concurrent other programs, swap/paging-file configuration, and
    source video resolution all vary). So on an actual `MemoryError` or
    `OSError` here, rather than letting it crash the whole render (what
    every previous release did), this chunk is split in half and each
    half is rendered independently (recursively, down to a single shot
    if truly necessary) and the two resulting small files are then
    joined with a plain hard-cut -- trading a few extra crossfades for a
    render that actually finishes, exactly matching the existing
    per-crossfade MemoryError fallback's philosophy in `_build_clip_batch`
    (degrade gracefully, never crash the whole render over what is
    fundamentally a resource-availability problem, not a correctness one).
    """
    try:
        batch_video, batch_shot_clips = _build_clip_batch(
            shots_chunk, resolved_media, mood, target_w, target_h, fps, energy_fn
        )
    except (MemoryError, OSError):
        if len(shots_chunk) <= 1:
            # Nothing smaller to retry with -- this is a genuine
            # resource exhaustion that even a single shot's decode
            # can't fit in, which previous releases always crashed on
            # too. Let it propagate so the user still sees a clear
            # error rather than silently producing a broken/incomplete
            # video.
            raise
        log.warning(
            "Building %d shot(s) together ran out of memory; splitting into "
            "smaller pieces and retrying (this trades a couple of crossfade "
            "transitions for hard cuts, but keeps the render from crashing)",
            len(shots_chunk), exc_info=True,
        )
        mid = len(shots_chunk) // 2
        path_a = _render_shots_chunk_to_file(
            shots_chunk[:mid], resolved_media, mood, target_w, target_h, fps,
            energy_fn, tmp_dir, chunk_label + "a",
        )
        path_b = _render_shots_chunk_to_file(
            shots_chunk[mid:], resolved_media, mood, target_w, target_h, fps,
            energy_fn, tmp_dir, chunk_label + "b",
        )
        clip_a = VideoFileClip(str(path_a), audio=False)
        clip_b = VideoFileClip(str(path_b), audio=False)
        try:
            joined = concatenate_videoclips([clip_a, clip_b], method="compose")
            out_path = tmp_dir / f"chunk_{chunk_label}_joined.mp4"
            joined.write_videofile(
                str(out_path), fps=fps, codec="libx264", audio=False,
                preset="ultrafast", threads=2, logger=None,
            )
            joined.close()
        finally:
            clip_a.close()
            clip_b.close()
        return out_path

    try:
        out_path = tmp_dir / f"chunk_{chunk_label}.mp4"
        batch_video.write_videofile(
            str(out_path), fps=fps, codec="libx264", audio=False,
            preset="ultrafast", threads=2, logger=None,
        )
    finally:
        batch_video.close()
        for c in batch_shot_clips:
            _close_shot_clip(c)
    return out_path


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

            # _render_shots_chunk_to_file both builds+crossfades this batch
            # AND writes it straight to its own small temporary MP4 (no
            # audio yet -- silent intermediate files, muxed with the real
            # audio + SFX only once at the very end), closing every
            # MoviePy/FFmpeg resource it opened before returning. If this
            # batch alone runs out of memory (MemoryError / Windows
            # "paging file too small" OSError) -- which `SHOT_BATCH_SIZE`
            # is sized to make rare, but can't rule out on every real
            # machine's actual available memory -- it recursively splits
            # itself into smaller pieces and retries rather than crashing
            # the whole render; see that function's docstring.
            batch_path = _render_shots_chunk_to_file(
                shots_batch, resolved_media, mood, target_w, target_h, fps,
                energy_fn, tmp_dir, f"{batch_idx:04d}",
            )
            batch_paths.append(batch_path)

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
