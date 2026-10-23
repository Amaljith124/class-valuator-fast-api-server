"""Remove speech-recognition noise (sound tags, repetition loops, stray scripts) from a transcript.

Whisper often hallucinates when it hears music, buzzing or silence: the same phrase repeated dozens
of times, or text in a language nobody spoke. Left in, this noise wastes tokens and makes the
scoring model judge the recognition errors instead of the teaching.
"""

import re
import unicodedata
from collections import Counter

_TAGS = re.compile(r"\[[^\]]*\]|\([^)]*\)|[♪♫]+")
_PUNCTUATION = re.compile(r"[^\w]+", re.UNICODE)
MAX_LOOP_UNIT_WORDS = 12
# A short unit (one or two words) must repeat more often than a longer one to count as a loop,
# so ordinary emphasis such as "yes yes yes" is kept.
_MIN_REPEATS_SHORT_UNIT = 4
_MIN_REPEATS_LONG_UNIT = 3
# A script that makes up less than this share of all letters is treated as a recognition artefact.
MINOR_SCRIPT_SHARE = 0.10
STRAY_SCRIPT_DOMINANCE = 0.8
MIN_DUPLICATE_SEGMENT_RUN = 3


def _normalise_word(word: str) -> str:
    return _PUNCTUATION.sub("", word.casefold())


def _repeats(words: list[str], start: int, unit: int) -> int:
    """How many times words[start:start + unit] repeats back to back."""
    pattern = words[start:start + unit]
    count = 0
    position = start
    while words[position:position + unit] == pattern and len(pattern) == unit:
        count += 1
        position += unit
    return count


def remove_loops(text: str) -> tuple[str, int]:
    """Cut out runs where the same word or phrase repeats; keep the speech around them."""
    tokens = text.split()
    removed = 0
    while True:
        words = [_normalise_word(token) for token in tokens]
        found = None
        for start in range(len(words)):
            if not words[start]:
                continue
            for unit in range(1, MAX_LOOP_UNIT_WORDS + 1):
                needed = _MIN_REPEATS_SHORT_UNIT if unit <= 2 else _MIN_REPEATS_LONG_UNIT
                repeats = _repeats(words, start, unit)
                if repeats >= needed:
                    found = (start, start + repeats * unit)
                    break
            if found:
                break
        if not found:
            return " ".join(tokens), removed
        tokens = tokens[:found[0]] + tokens[found[1]:]
        removed += 1


def _script(character: str) -> str | None:
    if not character.isalpha():
        return None
    try:
        return unicodedata.name(character).split()[0]
    except ValueError:
        return None


def _script_counts(text: str) -> Counter:
    return Counter(script for script in map(_script, text) if script)


def clean_transcript(transcript: dict) -> dict:
    """Return the transcript without recognition noise, plus a `cleaning` summary."""
    segments = transcript.get("segments") or []
    stats = {"segments_in": len(segments), "tags_removed": 0, "loops_removed": 0,
             "stray_script_removed": 0, "duplicates_removed": 0, "empty_removed": 0}

    stage_one = []
    for segment in segments:
        text = str(segment.get("text") or "")
        without_tags = _TAGS.sub(" ", text)
        if without_tags != text:
            stats["tags_removed"] += 1
        without_loops, loops = remove_loops(without_tags)
        stats["loops_removed"] += loops
        stage_one.append({**segment, "text": " ".join(without_loops.split())})

    totals = Counter()
    for segment in stage_one:
        totals.update(_script_counts(segment["text"]))
    all_letters = sum(totals.values())

    cleaned = []
    previous_text = None
    run = 0
    for segment in stage_one:
        text = segment["text"]
        counts = _script_counts(text)
        letters = sum(counts.values())
        if letters == 0:
            stats["empty_removed"] += 1
            continue
        top_script, top_count = counts.most_common(1)[0]
        if (
            all_letters
            and top_count / letters >= STRAY_SCRIPT_DOMINANCE
            and totals[top_script] / all_letters < MINOR_SCRIPT_SHARE
        ):
            stats["stray_script_removed"] += 1
            continue
        key = _normalise_word(text.replace(" ", ""))
        run = run + 1 if key == previous_text else 1
        previous_text = key
        if run >= MIN_DUPLICATE_SEGMENT_RUN and len(key) >= 3:
            stats["duplicates_removed"] += 1
            continue
        cleaned.append(segment)

    stats["segments_out"] = len(cleaned)
    return {
        **transcript,
        "segments": cleaned,
        "text": " ".join(segment["text"] for segment in cleaned),
        "cleaning": stats,
    }
