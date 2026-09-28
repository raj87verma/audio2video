"""Optional local speech-to-text transcription + keyword extraction.

Uses faster-whisper (open-source, MIT licensed, runs the Whisper model
fully locally via CTranslate2 — no API key, no per-request cost, no data
leaves the machine). This lets the pipeline pull search keywords straight
out of a song's lyrics/vocals when present, so fetched stock footage can
match what's actually being sung/said instead of only the overall mood.

If no vocals are present, or transcription confidence is too low, the
caller should simply fall back to the mood-derived keywords — this module
never raises for "no speech found", it just returns an empty result.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# A small built-in stopword list — enough to filter filler words out of
# lyrics without pulling in a heavyweight NLP dependency.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "so",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "ours", "theirs",
    "this", "that", "these", "those",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under", "again", "further",
    "not", "no", "nor", "just", "don", "now", "there", "here",
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "oh", "yeah", "yea", "na", "la", "ooh", "uh", "ah", "hey", "gonna", "wanna",
    "got", "get", "getting", "like", "know", "cause", "cuz",
    "all", "some", "such", "only", "own", "same", "too", "very", "s", "t", "m",
    "re", "ve", "ll", "d",
}


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    full_text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = "unknown"
    has_speech: bool = False

    def keywords_in(self, start: float, end: float, limit: int = 3) -> list[str]:
        """Extract salient keywords spoken/sung between start and end seconds."""
        words: list[str] = []
        for seg in self.segments:
            # Require actual overlap (not just touching boundaries) so a
            # segment ending exactly at `start` (or starting exactly at
            # `end`) isn't double-counted in two adjacent shot windows.
            if seg.end <= start or seg.start >= end:
                continue
            words.extend(_extract_keywords(seg.text))
        # de-duplicate while preserving order
        seen = set()
        unique = []
        for w in words:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique[:limit]


def _extract_keywords(text: str, limit: int = 8) -> list[str]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    keywords = [w for w in words if len(w) > 2 and w not in _STOPWORDS]
    return keywords[:limit]


def transcribe_audio(
    path: str,
    model_size: str = "base",
    enabled: bool = True,
) -> TranscriptResult:
    """Transcribe vocals/speech in an audio file using faster-whisper.

    Runs fully locally (downloads the Whisper model weights once from
    Hugging Face on first use — free, open weights — then works offline).
    Returns an empty TranscriptResult (has_speech=False) if disabled, if the
    model can't be loaded, or if no speech is detected, so callers can
    always safely fall back to mood-based keywords without special-casing
    errors.
    """
    if not enabled:
        return TranscriptResult(full_text="", has_speech=False)

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover - optional dependency
        log.warning("faster-whisper not available (%s); skipping transcription", exc)
        return TranscriptResult(full_text="", has_speech=False)

    try:
        log.info("Loading whisper model '%s' (first run downloads weights)...", model_size)
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments_gen, info = model.transcribe(path, beam_size=1, vad_filter=True)

        segments: list[TranscriptSegment] = []
        full_text_parts: list[str] = []
        for seg in segments_gen:
            text = seg.text.strip()
            if not text:
                continue
            segments.append(TranscriptSegment(start=seg.start, end=seg.end, text=text))
            full_text_parts.append(text)

        full_text = " ".join(full_text_parts).strip()
        has_speech = len(full_text) > 0
        log.info(
            "Transcription complete: language=%s segments=%d has_speech=%s",
            getattr(info, "language", "unknown"), len(segments), has_speech,
        )
        return TranscriptResult(
            full_text=full_text,
            segments=segments,
            language=getattr(info, "language", "unknown"),
            has_speech=has_speech,
        )
    except Exception as exc:
        log.warning("Transcription failed (%s); continuing without lyrics", exc)
        return TranscriptResult(full_text="", has_speech=False)


def top_keywords(result: TranscriptResult, limit: int = 10) -> list[str]:
    """Most frequent non-stopword keywords across the whole transcript."""
    if not result.has_speech:
        return []
    from collections import Counter

    words = _extract_keywords(result.full_text, limit=10_000)
    counts = Counter(words)
    return [w for w, _ in counts.most_common(limit)]
