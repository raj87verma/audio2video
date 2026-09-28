"""Rule-based mood / genre-feel classifier.

No external API or ML model is required: we derive a mood label and a set
of visual search keywords purely from the signal features already computed
by `audio_analysis.analyze_audio`. This keeps the whole pipeline free,
offline-capable and deterministic.

The classifier looks at:
  - tempo (BPM)              -> energetic vs calm
  - loudness_mean / std      -> intensity & dynamic range
  - harmonic_ratio           -> melodic/ambient vs percussive/rhythmic
  - spectral_centroid_mean   -> bright/sharp vs warm/dark timbre
  - zero_crossing_rate_mean  -> noisiness (distortion, aggressive energy)

Moods map to:
  - a human-readable label (shown in the UI)
  - stock-media search keywords (fed to Pexels/Pixabay)
  - a color-grading preset (used by the video builder)
  - a cut-speed multiplier (shorter shots for high energy moods)
"""
from __future__ import annotations

from dataclasses import dataclass

from .audio_analysis import AudioFeatures


@dataclass
class MoodProfile:
    label: str
    keywords: list[str]
    color_grade: str          # name of a LUT-like preset applied in video_builder
    cut_speed: float          # multiplier on base shot duration (1.0 = normal, <1 = faster cuts)
    description: str


# Ordered rule table. First matching rule wins. Each rule is a predicate over
# the AudioFeatures plus a MoodProfile factory.
def _rules(f: AudioFeatures):
    bright = f.spectral_centroid_mean > 2200
    dark = f.spectral_centroid_mean < 1200
    noisy = f.zero_crossing_rate_mean > 0.09
    loud = f.loudness_mean > 0.55
    quiet = f.loudness_mean < 0.3
    wide_dynamics = f.loudness_std > 0.18
    percussive = f.harmonic_ratio < 0.45
    melodic = f.harmonic_ratio > 0.65

    return [
        # High tempo + loud + percussive/noisy -> aggressive / energetic action
        (
            f.tempo >= 135 and (loud or noisy),
            MoodProfile(
                label="High-Energy / Action",
                keywords=["fast action", "energy", "urban night", "sports action", "fireworks", "neon city"],
                color_grade="high_contrast_cool",
                cut_speed=0.55,
                description="Fast tempo, high loudness — quick cuts, punchy color grade.",
            ),
        ),
        # Medium-fast tempo, bright, melodic -> upbeat / happy / pop
        (
            f.tempo >= 100 and bright and not percussive,
            MoodProfile(
                label="Upbeat / Happy",
                keywords=["celebration", "sunshine", "friends laughing", "colorful city", "dance", "summer"],
                color_grade="warm_vibrant",
                cut_speed=0.75,
                description="Bright timbre and lively tempo — warm, vibrant, cheerful visuals.",
            ),
        ),
        # Slow tempo, dark timbre, wide dynamics, melodic -> emotional / dramatic ballad
        (
            f.tempo < 90 and (dark or wide_dynamics) and melodic,
            MoodProfile(
                label="Emotional / Dramatic",
                keywords=["rain window", "silhouette sunset", "lonely road", "candle light", "storm clouds", "slow motion portrait"],
                color_grade="moody_desaturated",
                cut_speed=1.6,
                description="Slow, dark and melodic — long, cinematic, emotionally weighted shots.",
            ),
        ),
        # Slow tempo, quiet, melodic, not much percussion -> calm / ambient / chill
        (
            f.tempo < 95 and (quiet or melodic) and not noisy,
            MoodProfile(
                label="Calm / Ambient",
                keywords=["ocean waves", "forest morning", "clouds timelapse", "soft light nature", "meditation", "slow river"],
                color_grade="soft_pastel",
                cut_speed=2.0,
                description="Low intensity, melodic — long, tranquil, slow-moving shots.",
            ),
        ),
        # Percussive & mid tempo, not particularly bright or dark -> rhythmic / groove
        (
            percussive and 90 <= f.tempo < 135,
            MoodProfile(
                label="Rhythmic / Groove",
                keywords=["street dance", "city lights night", "abstract geometric", "drums closeup", "crowd concert"],
                color_grade="high_contrast_cool",
                cut_speed=0.85,
                description="Strong rhythm, moderate tempo — beat-synced, punchy cuts.",
            ),
        ),
    ]


def classify_mood(features: AudioFeatures) -> MoodProfile:
    """Return the best-matching MoodProfile for the given audio features.

    Falls back to a sensible generic "Balanced / Cinematic" profile if none
    of the specific rules match, so the caller always gets a usable result.
    """
    for matched, profile in _rules(features):
        if matched:
            return profile

    return MoodProfile(
        label="Balanced / Cinematic",
        keywords=["cinematic landscape", "city skyline", "nature wide shot", "abstract background", "travel"],
        color_grade="neutral_cinematic",
        cut_speed=1.0,
        description="No strong signal in a specific direction — balanced, general-purpose visuals.",
    )
