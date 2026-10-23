import re

from app.services.teaching_service import CRITERIA

# Weighted toward how the lesson is taught (explanation, methods, structure) rather than how
# correct the content is, which good and weak classes tend to share.
WEIGHTS = {
    "subject_accuracy": 0.12,
    "explanation_clarity": 0.20,
    "teaching_methods": 0.16,
    "class_structure": 0.12,
    "pacing": 0.10,
    "student_interaction": 0.10,
    "responsiveness": 0.10,
    "lesson_summary": 0.10,
}

_QUOTE_KEYS = ("quote", "text", "excerpt", "evidence")
_ELLIPSIS = re.compile(r"\.{3,}|…")
_NON_WORD = re.compile(r"[\W_]+")


def _normalize_quote(value: str) -> str:
    """Case, whitespace and punctuation-insensitive form used to compare quotes."""
    return _NON_WORD.sub(" ", value.casefold()).strip()


def _item_quote(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in _QUOTE_KEYS:
            if isinstance(item.get(key), str):
                return item[key]
    return ""


def _valid_quote(evidence, transcript: str) -> bool:
    """True only if at least one quote is present and every quoted fragment is in the transcript."""
    if not isinstance(evidence, list) or not evidence:
        return False
    normalized_transcript = _normalize_quote(transcript)
    verified_any = False
    for item in evidence:
        quote = _item_quote(item)
        # A quote may skip words with "..."; each fragment must still appear verbatim.
        fragments = [fragment for fragment in (_normalize_quote(part) for part in _ELLIPSIS.split(quote)) if fragment]
        for fragment in fragments:
            if fragment not in normalized_transcript:
                return False
            verified_any = True
    return verified_any


def combine(teaching: dict, audio: dict, video: dict, transcript: dict) -> dict:
    text = transcript.get("text", "")
    teaching_criteria = teaching.get("criteria")
    if not isinstance(teaching_criteria, dict):
        teaching_criteria = {}
    validated = {}
    total = 0.0
    used = 0.0

    for name in CRITERIA:
        item = teaching_criteria.get(name)
        if not isinstance(item, dict):
            item = {}
        score = item.get("score")
        assessable = (
            bool(item.get("assessable"))
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
        )
        reason = item.get("reason_unassessable") or "Insufficient transcript evidence"

        if assessable and not 0 <= score <= 100:
            assessable = False
            reason = "Score outside allowed range"

        evidence = item.get("evidence", [])
        if assessable and not evidence:
            assessable = False
            reason = "No supporting transcript evidence was provided"
        elif assessable and not _valid_quote(evidence, text):
            assessable = False
            reason = "Evidence quote could not be verified in transcript"

        score = float(score) if assessable else None
        validated[name] = {
            "score": score,
            "evidence": evidence if isinstance(evidence, list) else [],
            "assessable": assessable,
            "reason_unassessable": None if assessable else reason,
            "score_notes": item.get("score_notes", []) if assessable and isinstance(item.get("score_notes"), list) else [],
        }
        if assessable:
            total += score * WEIGHTS[name]
            used += WEIGHTS[name]

    return {
        "criteria": validated,
        "overall_score": round(total / used, 2) if used else None,
        "score_coverage": round(used, 3),
        "weights": WEIGHTS,
        "lesson_observations": teaching.get("observations", {}) if isinstance(teaching.get("observations"), dict) else {},
        "audio_quality": audio,
        "video_screen_analysis": video,
        "summary": teaching.get("lesson_summary", ""),
        "strengths": teaching.get("strengths", []),
        "improvements": teaching.get("improvements", []),
        "scoring_notes": [
            "Weighted mean over assessable teaching criteria only.",
            "Audio and video results are separate from the teaching score.",
        ],
    }
