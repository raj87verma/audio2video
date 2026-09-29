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
states the theme in words a stock-media search can actually use.

Generic stock-media libraries (Pexels/Pixabay) have essentially zero
coverage of specific Hindu deities -- searching "Khatu Shyam" on either
returns nothing, so even a perfect keyword extraction can only ever get
generic "temple"/"prayer"/"diya" visuals from those sources, not the
actual named deity a song is about. Wikimedia Commons, by contrast, is
crowd-sourced by devotees photographing/filming their own temples and
ceremonies, and *does* have real, specific, on-theme coverage (verified:
"Khatu Shyam" alone returns 49 real hits including temple photos and the
deity's own murti; "aarti filetype:video" returns real ceremony footage).
So this module extracts two distinct things:

  - `is_devotional` / `keywords`: the existing generic devotional-theme
    signal (unchanged), used for Pexels/Pixabay queries and as the final
    fallback when no specific deity is identified.
  - `deity` / `deity_search_terms`: a *specific* recognized deity/entity
    name when one is present, with a couple of Wikimedia-tuned query
    variants (e.g. "Khatu Shyam", "Krishna idol") that were verified to
    return relevant, on-theme results rather than noise (a bare "Krishna"
    query, for instance, also matches an unrelated butterfly species and
    a temple in Singapore named after a different deity; qualifying with
    "idol"/"murti"/"temple" reliably filters that out for common deities).

This module deliberately does NOT try to be a general-purpose topic
classifier — it only recognizes a curated set of devotional/spiritual
terms and named deities (covering Hindi devotional vocabulary in both
Devanagari and common English transliteration, plus a few other
traditions) since that's the concrete case reported. Filenames/lyrics
that don't match anything simply fall through to the existing acoustic
mood classifier unchanged.
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
    # A specific recognized deity/entity name (e.g. "Khatu Shyam",
    # "Hanuman"), or None if the filename/lyrics only matched generic
    # devotional vocabulary without naming a particular deity. Used to
    # drive a Wikimedia Commons search for the *actual named subject*
    # instead of only generic devotional stock-media keywords.
    deity: str | None = None
    # A handful of Wikimedia-Commons-tuned query strings for `deity`,
    # ordered from most to least specific. Bare deity names often pull in
    # unrelated noise (e.g. "Krishna" alone also matches an unrelated
    # butterfly species and places merely named after the deity), so most
    # entries are qualified with "idol"/"murti"/"temple"/"aarti" -- see
    # this module's docstring for verified search-quality notes. Empty
    # when `deity` is None.
    deity_search_terms: list[str] = field(default_factory=list)


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

# Maps every romanized alias token that can identify a specific deity to
# a canonical (display_name, [wikimedia_search_terms...]) entry. Multiple
# aliases (e.g. "shiv"/"shiva"/"mahadev") intentionally map to the same
# canonical entry so any of the common spellings/epithets found in a
# filename resolve to the same, verified-good search terms.
#
# Search terms are ordered most-specific-first and were spot-checked
# against the real Wikimedia Commons search API (see this module's
# docstring and the PR description for example result counts/titles) to
# confirm they return genuinely on-theme photos/videos rather than noise.
# A bare deity name is deliberately avoided as the *only* term for
# deities with common-word or ambiguous names (Krishna, Ram, Kali, Devi)
# since those alone pull in unrelated matches (a butterfly species, a
# temple merely named after a different deity, etc.) -- qualifying with
# "idol"/"murti"/"temple"/"aarti" was verified to filter that out.
_DEITY_ALIASES: dict[str, tuple[str, list[str]]] = {
    # Khatu Shyam -- the concrete case reported; bare "Khatu Shyam" is
    # already specific enough (49 real hits, verified no noise).
    "khatu": ("Khatu Shyam", ["Khatu Shyam", "Khatu Shyam temple", "Khatu Shyam aarti"]),
    "shyam": ("Khatu Shyam", ["Khatu Shyam", "Khatu Shyam temple", "Khatu Shyam aarti"]),
    "shyamji": ("Khatu Shyam", ["Khatu Shyam", "Khatu Shyam temple", "Khatu Shyam aarti"]),

    "krishna": ("Krishna", ["Krishna idol", "Krishna murti", "Radha Krishna temple"]),
    "krishn": ("Krishna", ["Krishna idol", "Krishna murti", "Radha Krishna temple"]),
    "kanha": ("Krishna", ["Krishna idol", "Krishna murti", "Radha Krishna temple"]),
    "radha": ("Radha Krishna", ["Radha Krishna idol", "Radha Krishna temple"]),

    "ram": ("Ram", ["Ram idol", "Ram temple", "Ram Sita temple"]),
    "rama": ("Ram", ["Ram idol", "Ram temple", "Ram Sita temple"]),
    "raam": ("Ram", ["Ram idol", "Ram temple", "Ram Sita temple"]),
    "sita": ("Sita Ram", ["Ram Sita temple", "Sita Ram idol"]),

    "hanuman": ("Hanuman", ["Hanuman idol", "Hanuman temple", "Bajrangbali idol"]),
    "bajrangbali": ("Hanuman", ["Hanuman idol", "Hanuman temple", "Bajrangbali idol"]),

    "shiv": ("Shiva", ["Shiva idol", "Shiva temple", "Mahadev idol"]),
    "shiva": ("Shiva", ["Shiva idol", "Shiva temple", "Mahadev idol"]),
    "mahadev": ("Shiva", ["Shiva idol", "Shiva temple", "Mahadev idol"]),

    "durga": ("Durga", ["Durga idol", "Durga Puja", "Durga temple"]),
    "kali": ("Kali", ["Kali idol", "Kali Puja", "Kali temple"]),
    "amba": ("Amba Mata", ["Amba Mata temple", "Ambe Maa idol"]),
    "ambe": ("Amba Mata", ["Amba Mata temple", "Ambe Maa idol"]),

    "lakshmi": ("Lakshmi", ["Lakshmi idol", "Lakshmi Puja", "Lakshmi temple"]),
    "laxmi": ("Lakshmi", ["Lakshmi idol", "Lakshmi Puja", "Lakshmi temple"]),
    "saraswati": ("Saraswati", ["Saraswati idol", "Saraswati Puja", "Saraswati temple"]),

    "ganesh": ("Ganesh", ["Ganesh idol", "Ganesh murti", "Ganesh temple"]),
    "ganesha": ("Ganesh", ["Ganesh idol", "Ganesh murti", "Ganesh temple"]),
    "ganpati": ("Ganesh", ["Ganesh idol", "Ganesh murti", "Ganesh temple"]),

    "vishnu": ("Vishnu", ["Vishnu idol", "Vishnu temple"]),
    "narayan": ("Vishnu", ["Vishnu idol", "Vishnu temple"]),

    "sai": ("Sai Baba", ["Sai Baba idol", "Sai Baba temple", "Shirdi Sai Baba"]),
    "balaji": ("Balaji", ["Balaji temple", "Tirupati Balaji idol"]),
    "vaishno": ("Vaishno Devi", ["Vaishno Devi", "Vaishno Devi temple"]),
}

# Devanagari aliases map through the same canonical entries. Kept as a
# separate dict (rather than merged into _DEITY_ALIASES) purely so the
# romanized table above stays easy to scan; both are consulted together
# in `_deity_from_tokens`.
_DEITY_ALIASES_DEVANAGARI: dict[str, tuple[str, list[str]]] = {
    "खाटू": _DEITY_ALIASES["khatu"],
    "श्याम": _DEITY_ALIASES["shyam"],
    "कृष्ण": _DEITY_ALIASES["krishna"],
    "राधा": _DEITY_ALIASES["radha"],
    "राम": _DEITY_ALIASES["ram"],
    "सीता": _DEITY_ALIASES["sita"],
    "हनुमान": _DEITY_ALIASES["hanuman"],
    "शिव": _DEITY_ALIASES["shiv"],
    "दुर्गा": _DEITY_ALIASES["durga"],
    "काली": _DEITY_ALIASES["kali"],
    "लक्ष्मी": _DEITY_ALIASES["lakshmi"],
    "सरस्वती": _DEITY_ALIASES["saraswati"],
    "गणेश": _DEITY_ALIASES["ganesh"],
    "विष्णु": _DEITY_ALIASES["vishnu"],
    "नारायण": _DEITY_ALIASES["narayan"],
    "साई": _DEITY_ALIASES["sai"],
    "बालाजी": _DEITY_ALIASES["balaji"],
}


def _deity_from_tokens(tokens: set[str]) -> tuple[str | None, list[str]]:
    """Return (canonical_deity_name, search_terms) for the first deity
    alias found in `tokens`, preferring the most specific match.

    Iteration order over a Python dict follows insertion order, and
    _DEITY_ALIASES is intentionally written with "khatu"/"shyam" (the
    single most specific case handled) first, so a filename mentioning
    both a specific deity and a generic one resolves to the specific one.
    """
    for alias, (name, terms) in _DEITY_ALIASES.items():
        if alias in tokens:
            return name, terms
    for alias, (name, terms) in _DEITY_ALIASES_DEVANAGARI.items():
        if alias in tokens:
            return name, terms
    return None, []


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

    When a specific deity/entity is additionally identified (e.g. "Khatu
    Shyam"), `deity` and `deity_search_terms` are populated too --
    callers should prefer searching those against Wikimedia Commons
    before falling back to the generic `keywords` against Pexels/Pixabay,
    since generic stock-media libraries have essentially no coverage of
    specific deities.
    """
    combined = f"{filename} {transcript_text}"
    tokens = _tokenize(combined)

    matched = sorted((tokens & _DEVOTIONAL_TERMS) | (tokens & _DEVOTIONAL_TERMS_DEVANAGARI))
    deity, deity_terms = _deity_from_tokens(tokens)

    if not matched and not deity:
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
        deity=deity,
        deity_search_terms=deity_terms,
    )
