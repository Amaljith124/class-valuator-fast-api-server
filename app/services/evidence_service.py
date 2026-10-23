from app.services.teaching_service import CRITERIA

WEIGHTS = {
    "subject_accuracy": 0.20,
    "explanation_clarity": 0.15,
    "teaching_methods": 0.12,
    "class_structure": 0.10,
    "pacing": 0.10,
    "student_interaction": 0.13,
    "responsiveness": 0.10,
    "lesson_summary": 0.10,
}


def _normalize_quote(value: str) -> str:
    return " ".join(value.casefold().split())


def _valid_quote(evidence, transcript: str) -> bool:
    if not evidence:
        return False
    normalized_transcript = _normalize_quote(transcript)
    for item in evidence:
        quote = item if isinstance(item, str) else item.get("quote", "") if isinstance(item, dict) else ""
        if quote and _normalize_quote(quote) not in normalized_transcript:
            return False
    return True


def combine(teaching: dict, audio: dict, video: dict, transcript: dict) -> dict:
    text = transcript.get("text", "")
    validated = {}
    total = 0.0
    used = 0.0

    for name in CRITERIA:
        item = teaching.get("criteria", {}).get(name, {})
        score = item.get("score")
        assessable = (
            bool(item.get("assessable"))
            and isinstance(score, (int, float))
            and not isinstance(score, bool)
        )
        if assessable and not 0 <= score <= 100:
            raise ValueError("Teaching score outside allowed range")

        evidence = item.get("evidence", [])
        quotes_verified = isinstance(evidence, list) and _valid_quote(evidence, text)
        if assessable and not quotes_verified:
            assessable = False
            reason = "Evidence quote could not be verified in transcript"
        else:
            reason = item.get("reason_unassessable", "Insufficient transcript evidence")

        assessable = assessable and bool(evidence)
        score = float(score) if assessable else None
        validated[name] = {
            "score": score,
            "evidence": evidence if isinstance(evidence, list) else [],
            "assessable": assessable,
            "reason_unassessable": None if assessable else reason,
        }
        if assessable:
            total += score * WEIGHTS[name]
            used += WEIGHTS[name]

    return {
        "criteria": validated,
        "overall_score": round(total / used, 2) if used else None,
        "score_coverage": round(used, 3),
        "weights": WEIGHTS,
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
