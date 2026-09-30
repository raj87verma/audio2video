"""Sound-effect generation.

Procedurally generated whoosh / impact / riser / sparkle sounds (numpy
only, zero network access) used as beat-synced transition and highlight
cues. These are what give the final video its "SFX added automatically"
feel entirely offline.

All synthesized effects are returned as float32 numpy arrays (mono, at the
given sample rate) so the video builder can mix them directly with
MoviePy's AudioArrayClip without needing any file I/O.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic SFX generators
# ---------------------------------------------------------------------------


def _fade(signal: np.ndarray, sr: int, fade_in: float = 0.0, fade_out: float = 0.0) -> np.ndarray:
    n = len(signal)
    out = signal.copy()
    if fade_in > 0:
        fi = min(int(fade_in * sr), n)
        out[:fi] *= np.linspace(0, 1, fi)
    if fade_out > 0:
        fo = min(int(fade_out * sr), n)
        out[-fo:] *= np.linspace(1, 0, fo)
    return out


def generate_whoosh(duration: float = 0.6, sr: int = 44100, rising: bool = True) -> np.ndarray:
    """Filtered-noise 'whoosh' — good for transitions between shots."""
    n = int(duration * sr)
    noise = np.random.default_rng(abs(hash(("whoosh", duration))) % (2**32)).normal(0, 1, n)

    # Sweep a bandpass center frequency to create the "whoosh" motion feel.
    f_start, f_end = (400, 4000) if rising else (4000, 400)
    center = np.linspace(f_start, f_end, n)
    # simple time-varying resonant filter approximated via modulated sine-weighted noise
    carrier = np.sin(2 * np.pi * np.cumsum(center) / sr)
    signal = noise * (0.5 + 0.5 * carrier)

    # amplitude envelope: quick attack, smooth decay
    env = np.concatenate([
        np.linspace(0, 1, max(1, n // 6)) ** 2,
        np.linspace(1, 0, n - max(1, n // 6)) ** 1.5,
    ])[:n]
    signal = signal * env
    signal = _fade(signal, sr, fade_in=0.01, fade_out=0.05)
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.6
    return signal.astype(np.float32)


def generate_impact(duration: float = 0.4, sr: int = 44100, punch: float = 1.0) -> np.ndarray:
    """Low-frequency thump + transient click — good for hard cuts / beats."""
    n = int(duration * sr)
    t = np.linspace(0, duration, n, endpoint=False)

    # Sub-bass thump with pitch drop (classic kick-drum synthesis trick).
    freq_env = 120 * punch * np.exp(-t * 18)
    phase = 2 * np.pi * np.cumsum(freq_env) / sr
    thump = np.sin(phase) * np.exp(-t * 9)

    # Transient click (broadband noise burst at the very start).
    click_len = max(1, int(0.01 * sr))
    click = np.zeros(n)
    rng = np.random.default_rng(abs(hash(("impact", duration, punch))) % (2**32))
    click[:click_len] = rng.normal(0, 1, click_len) * np.linspace(1, 0, click_len)

    signal = thump * 0.8 + click * 0.5
    signal = _fade(signal, sr, fade_in=0.001, fade_out=0.05)
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.7
    return signal.astype(np.float32)


def generate_riser(duration: float = 1.5, sr: int = 44100) -> np.ndarray:
    """Rising pitch/noise swell — good for building tension into a highlight."""
    n = int(duration * sr)

    freq = np.linspace(150, 2500, n)
    tone = np.sin(2 * np.pi * np.cumsum(freq) / sr)

    rng = np.random.default_rng(abs(hash(("riser", duration))) % (2**32))
    noise = rng.normal(0, 1, n)
    noise_env = np.linspace(0, 1, n) ** 2

    env = np.linspace(0.05, 1.0, n) ** 1.5
    signal = (tone * 0.6 + noise * noise_env * 0.4) * env
    signal = _fade(signal, sr, fade_in=0.05, fade_out=0.02)
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.5
    return signal.astype(np.float32)


def generate_sparkle(duration: float = 0.5, sr: int = 44100) -> np.ndarray:
    """Bright high-frequency shimmer — good for gentle/emotional highlight accents."""
    n = int(duration * sr)
    t = np.linspace(0, duration, n, endpoint=False)
    rng = np.random.default_rng(abs(hash(("sparkle", duration))) % (2**32))

    freqs = rng.uniform(2000, 6000, 6)
    signal = np.zeros(n)
    for f in freqs:
        phase = rng.uniform(0, 2 * np.pi)
        signal += np.sin(2 * np.pi * f * t + phase)
    signal /= len(freqs)

    env = np.exp(-t * 6)
    signal = signal * env
    signal = _fade(signal, sr, fade_in=0.005, fade_out=0.05)
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.4
    return signal.astype(np.float32)


_SYNTH_BY_MOOD_LEVEL = {
    ("high", "transition"): lambda sr: generate_whoosh(0.5, sr, rising=True),
    ("high", "highlight"): lambda sr: generate_impact(0.4, sr, punch=1.2),
    ("medium", "transition"): lambda sr: generate_whoosh(0.4, sr, rising=True),
    ("medium", "highlight"): lambda sr: generate_impact(0.35, sr, punch=0.9),
    ("low", "transition"): lambda sr: generate_sparkle(0.5, sr),
    ("low", "highlight"): lambda sr: generate_sparkle(0.6, sr),
}


def synth_sfx_for(energy_level: str, purpose: str, sr: int = 44100) -> np.ndarray:
    """Pick+generate a synthetic SFX appropriate for a shot's energy/purpose.

    `purpose` is 'transition' (played at a cut) or 'highlight' (played on a
    strong onset inside a high-energy shot). Always returns a valid numpy
    array — unknown combos fall back to a generic whoosh.
    """
    key = (energy_level, purpose)
    factory = _SYNTH_BY_MOOD_LEVEL.get(key)
    if factory is None:
        factory = lambda sr: generate_whoosh(0.5, sr)
    return factory(sr)
