"""Procedural fallback visuals — generated entirely with PIL + numpy.

Used whenever a shot's stock-media search (Pexels/Pixabay) comes back
empty — no API key configured, rate-limited, no results for the keyword,
or a network hiccup. This guarantees the pipeline can ALWAYS produce a
finished video, even completely offline with zero API keys, by rendering
mood-colored animated gradient/particle backgrounds plus a live audio
waveform/spectrum readout synced to the music.

Everything here is deterministic-ish (seeded by shot index) so re-runs of
the same project look consistent, and cheap enough to generate per-frame
at render time without pre-rendering huge files to disk.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Color palettes per color_grade name (see mood.py). Each is a pair of RGB
# tuples used as a gradient's start/end colors.
_PALETTES = {
    "high_contrast_cool": [(10, 10, 30), (0, 200, 255), (255, 0, 120)],
    "warm_vibrant": [(255, 140, 0), (255, 60, 60), (255, 210, 0)],
    "moody_desaturated": [(20, 20, 25), (60, 50, 70), (90, 70, 60)],
    "soft_pastel": [(180, 220, 240), (230, 200, 230), (255, 240, 220)],
    "neutral_cinematic": [(15, 20, 25), (60, 80, 100), (120, 130, 140)],
}


def _palette_for(color_grade: str) -> list[tuple[int, int, int]]:
    return _PALETTES.get(color_grade, _PALETTES["neutral_cinematic"])


def _lerp_color(c1, c2, t: float) -> tuple[int, int, int]:
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def render_gradient_frame(
    width: int,
    height: int,
    color_grade: str,
    t: float,
    energy: float = 0.5,
    seed: int = 0,
) -> Image.Image:
    """Render one animated diagonal-gradient frame.

    `t` is a slow-moving phase (e.g. seconds since shot start) that animates
    the gradient direction/position; `energy` (0..1) modulates brightness
    and a subtle pulsing glow so the background visibly reacts to the music.
    """
    palette = _palette_for(color_grade)
    c1, c2 = palette[0], palette[1 % len(palette)]

    # Slow oscillating blend factor for a living, breathing gradient.
    phase = (math.sin(t * 0.3 + seed) + 1) / 2
    base = _lerp_color(c1, c2, phase)
    accent = palette[2 % len(palette)]

    # Build gradient via small numpy array then upscale — much faster than
    # per-pixel PIL drawing at full resolution.
    small_w, small_h = 64, 36
    yy, xx = np.mgrid[0:small_h, 0:small_w].astype(np.float32)
    angle = t * 0.15 + seed
    grad = (xx / small_w) * math.cos(angle) + (yy / small_h) * math.sin(angle)
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)

    brightness = 0.6 + 0.4 * energy
    r = (base[0] + (accent[0] - base[0]) * grad) * brightness
    g = (base[1] + (accent[1] - base[1]) * grad) * brightness
    b = (base[2] + (accent[2] - base[2]) * grad) * brightness

    arr = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB").resize((width, height), Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return img


def _draw_particles(draw: ImageDraw.ImageDraw, width: int, height: int, t: float, energy: float, seed: int, count: int = 40):
    rng = random.Random(seed)
    accent = (255, 255, 255)
    for i in range(count):
        base_x = rng.uniform(0, width)
        base_y = rng.uniform(0, height)
        speed = rng.uniform(10, 40) * (0.5 + energy)
        x = (base_x + t * speed) % width
        y = (base_y + math.sin(t * 0.5 + i) * 20) % height
        radius = rng.uniform(1, 3) * (0.6 + energy)
        alpha = int(60 + 120 * energy)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(*accent, alpha))


def render_particle_overlay(width: int, height: int, t: float, energy: float, seed: int = 0) -> Image.Image:
    """Transparent overlay with drifting particles — adds motion/VFX feel."""
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    _draw_particles(draw, width, height, t, energy, seed)
    return overlay


def render_waveform_bars(
    width: int,
    height: int,
    levels: np.ndarray,
    color: tuple[int, int, int] = (255, 255, 255),
    opacity: int = 160,
) -> Image.Image:
    """Render a simple bottom-anchored bar-style audio visualizer.

    `levels` is a 1D array of normalized (0..1) magnitudes, one per bar,
    typically a short recent slice of the RMS envelope around the current
    playback time.
    """
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    n = max(1, len(levels))
    bar_width = width / n
    max_bar_height = height * 0.25
    base_y = height - int(height * 0.08)

    for i, lvl in enumerate(levels):
        bar_h = max(2, int(lvl * max_bar_height))
        x0 = i * bar_width + bar_width * 0.15
        x1 = (i + 1) * bar_width - bar_width * 0.15
        y0 = base_y - bar_h
        y1 = base_y
        draw.rectangle([x0, y0, x1, y1], fill=(*color, opacity))
    return overlay


def compose_procedural_frame(
    width: int,
    height: int,
    color_grade: str,
    t: float,
    energy: float,
    recent_levels: np.ndarray | None = None,
    seed: int = 0,
) -> Image.Image:
    """Full procedural background: animated gradient + particles + optional
    waveform bars, composited into a single opaque RGB frame.

    This is what gets fed into the video builder as the "clip" for any shot
    whose stock-media fetch failed or was never attempted (no API keys).
    """
    frame = render_gradient_frame(width, height, color_grade, t, energy, seed).convert("RGBA")
    particles = render_particle_overlay(width, height, t, energy, seed)
    frame.alpha_composite(particles)

    if recent_levels is not None and len(recent_levels) > 0:
        bars = render_waveform_bars(width, height, recent_levels)
        frame.alpha_composite(bars)

    return frame.convert("RGB")
