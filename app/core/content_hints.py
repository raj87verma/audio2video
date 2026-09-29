"""Extracts content hints (devotional/religious theme + search keywords)
from an audio file's filename and, if available, its transcribed lyrics.

Why this exists: `mood.py` classifies mood purely from *acoustic* signal
features (tempo, loudness, timbre) — it has no idea what a song is
actually *about*. This works reasonably for generic pop/instrumental
tracks, but fails badly for devotional music: an aarti/bhajan can have a
tempo and brightness that acoustically resembles an "Upbeat / Happy" pop
song, so the mood classifier picks generic keywords like "celebration" or
"dance" — and Pexels/Pixabay dutifully return exactly that, with zero
connection to the actual devotional content.

The single most reliable, free, zero-latency signal for *what a song is
about* that we have almost every time is the file name itself: users
typically save devotional tracks with names like
"Khatu Shyam Ji Ki Aarti - Lakhbir Singh Lakkha.mp3", which plainly
states the theme in words a stock-media search can actually use ("aarti",
"temple", "prayer" all return real matching footage on Pexels/Pixabay,
even though a hyper-specific deity name like "Khatu Shyam" itself won't).

This module deliberately does NOT try to be a general-purpose topic
classifier — it only recognizes a curated set of devotional/spiritual
terms (covering Hindi devotional vocabulary in both Devanagari and common
English transliteration, plus a few other traditions) since that's the
concrete case reported. Filenames/lyrics that don't match anything simply
fall through to the existing acoustic mood classifier unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Devotional/spiritual vocabulary: English transliterations, Devanagari,
# and a handful of generic religious terms across traditions. Matched as
# whole "words" (see _tokenize) against filename + transcript text, so
# short substrings can't accidentally match unrelated words.
_DEVOTIONAL_TERMS = {
    # Generic devotional-format words -- strong signal on their own.
    "aarti", "arti", "aarthi", "bhajan", "bhajans", "kirtan", "kirtans",
    "satsang", "chalisa", "mantra", "mantras", "stuti", "stotram", "stotra",
    "puja", "pooja", "havan", "yagna", "prarthana", "vandana", "aradhana",
    # Places of worship / ritual objects.
    "mandir", "temple", "gurudwara", "masjid", "mosque", "church",
    "diya", "deepak", "aarti-diya", "incense", "agarbatti", "prasad",
    "prasadam", "ghanti",
    # Deities / revered figures (Hinduism-focused, matching the reported
    # case, but broad enough to cover common devotional uploads).
    "shyam", "krishna", "krishn", "kanha", "radha", "ram", "rama", "raam",
    "sita", "hanuman", "bajrangbali", "shiv", "shiva", "mahadev",
    "durga", "devi", "maa", "mata", "amba", "ambe", "kali", "lakshmi",
    "laxmi", "saraswati", "ganesh", "ganesha", "ganpati", "vishnu",
    "narayan", "sai", "khatu", "balaji", "vaishno", "shyamji",
    # Broad spiritual/religious descriptors.
    "devotional", "spiritual", "divine", "prayer", "prayers", "worship",
    "bhakti", "bhajan-sandhya", "jai", "namah", "namaha", "om", "aum",
    "guru", "sadhu", "sant", "sanatan",
}

# Devanagari-script devotional terms (transliteration above only catches
# romanized filenames; many devotional uploads keep the Devanagari title
# too, exactly as in the reported case: "खाटू श्याम जी की आरती").
_DEVOTIONAL_TERMS_DEVANAGARI = {
    "आरती", "भजन", "कीर्तन", "सत्संग", "चालीसा", "मंत्र", "पूजा", "मंदिर",
    "प्रार्थना", "श्याम", "कृष्ण", "राधा", "राम", "सीता", "हनुमान", "शिव",
    "दुर्गा", "देवी", "माता", "काली", "लक्ष्मी", "सरस्वती", "गणेश", "विष्णु",
    "नारायण", "साई", "खाटू", "बालाजी", "भक्ति", "जय", "गुरु", "संत", "दीप",
    "दीपक", "अगरबत्ती", "प्रसाद",
}


@dataclass
class ContentHints:
    is_devotional: bool = False
    matched_terms: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    # Multiplier applied on top of the mood's own cut_speed (see mood.py)
    # when set. Devotional/spiritual audio (aarti, bhajan, kirtan) is
    # paced for contemplative viewing, not fast-cut pop editing -- the
    # *acoustic* mood classifier has no way to know that from tempo/
    # loudness alone (a lively aarti can be acoustically indistinguishable
    # from an upbeat pop song), so this is applied as a content-level
    # override on top of whatever mood.py already picked. A value > 1.0
    # means longer, slower shots (fewer cuts overall); None means "don't
    # override, use the mood's own pacing".
    cut_speed_multiplier: float | None = None


# A small, curated set of stock-media-searchable keywords used whenever
# devotional content is detected. Chosen because they reliably return
# real, on-theme results on Pexels/Pixabay (unlike a specific deity name),
# while still visually matching the devotional/spiritual mood.
_DEVOTIONAL_KEYWORDS = [
    "hindu temple",
    "temple diya lamp",
    "incense smoke spiritual",
    "temple bells prayer",
    "sunrise temple silhouette",
    "flower offering prayer",
    "candle prayer spiritual",
    "meditation spiritual",
]


def _tokenize(text: str) -> set[str]:
    """Lowercased ASCII "word" tokens plus raw Devanagari word tokens.

    Devanagari doesn't have a-z case folding or word boundaries in the
    ASCII sense, so it's tokenized separately by matching contiguous
    Devanagari codepoints instead of relying on `\\w` behavior.
    """
    ascii_tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
    devanagari_tokens = set(re.findall(r"[\u0900-\u097F]+", text))
    return ascii_tokens | devanagari_tokens


def detect_content_hints(filename: str, transcript_text: str = "") -> ContentHints:
    """Detect a devotional/spiritual theme from a filename and/or transcript.

    Returns a `ContentHints` with `is_devotional=False` and empty keywords
    if nothing matches — callers should treat that as "no override, use
    the acoustic mood classifier's keywords as-is" rather than an error.
    """
    combined = f"{filename} {transcript_text}"
    tokens = _tokenize(combined)

    matched = sorted((tokens & _DEVOTIONAL_TERMS) | (tokens & _DEVOTIONAL_TERMS_DEVANAGARI))
    if not matched:
        return ContentHints()

    return ContentHints(
        is_devotional=True,
        matched_terms=matched,
        keywords=list(_DEVOTIONAL_KEYWORDS),
        # ~2.2x longer shots than the mood classifier alone would pick.
        # For a typical 6-minute aarti/bhajan this roughly halves the
        # total shot count (and therefore render time, since render cost
        # scales with shot count) on top of whatever the acoustic mood
        # already chose, while also matching the slower, more
        # contemplative visual pacing appropriate for devotional content.
        cut_speed_multiplier=2.2,
    )
