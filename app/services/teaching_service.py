import json
import math
import re

from app.services.gemini_service import generate_json
from app.templates.prompt_templates import TEACHING_EFFECTIVENESS_SCORING_INSTRUCTIONS

CRITERIA = ["subject_accuracy","explanation_clarity","teaching_methods","class_structure","pacing","student_interaction","responsiveness","lesson_summary"]
_CRITERION_ID_KEYS = ("id", "criterion_id", "criterion", "name", "key")


def _unassessable(reason: str) -> dict:
    return {"score": None, "evidence": [], "assessable": False, "reason_unassessable": reason}


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _criteria_by_name(raw_criteria) -> dict[str, dict]:
    """Accept criteria as {id: {...}} or as a list of {id: ..., ...} objects."""
    entries: dict[str, dict] = {}
    if isinstance(raw_criteria, dict):
        for key, value in raw_criteria.items():
            if isinstance(key, str) and isinstance(value, dict):
                entries.setdefault(_normalize_name(key), value)
    elif isinstance(raw_criteria, list):
        for item in raw_criteria:
            if not isinstance(item, dict):
                continue
            for id_key in _CRITERION_ID_KEYS:
                name = item.get(id_key)
                if isinstance(name, str) and name.strip():
                    entries.setdefault(_normalize_name(name), item)
                    break
    return entries


def _coerce_score(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip().rstrip("%").strip())
        except ValueError:
            return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _coerce_flag(value, default: bool) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    if value is None:
        return default
    return bool(value)


def _normalize_criterion(item: dict) -> dict:
    score = _coerce_score(item.get("score"))
    evidence = item.get("evidence")
    if isinstance(evidence, str):
        evidence = [evidence] if evidence.strip() else []
    elif not isinstance(evidence, list):
        evidence = []
    assessable = _coerce_flag(item.get("assessable"), default=score is not None) and score is not None
    reason = item.get("reason_unassessable") or item.get("reason")
    return {
        "score": score if assessable else None,
        "evidence": evidence,
        "assessable": assessable,
        "reason_unassessable": None if assessable else (
            str(reason) if reason else "Insufficient transcript evidence"
        ),
    }


def _string_list(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    items = []
    for entry in value:
        if isinstance(entry, dict):
            text = next(
                (entry[key] for key in ("text", "description", "point", "strength", "improvement") if isinstance(entry.get(key), str)),
                json.dumps(entry, ensure_ascii=False),
            )
        elif isinstance(entry, str):
            text = entry
        else:
            continue
        if text.strip():
            items.append(text.strip())
    return items


def _has_speech(transcript: dict) -> bool:
    return any(str(segment.get("text") or "").strip() for segment in transcript.get("segments", []))


def score(transcript: dict) -> dict:
    if not _has_speech(transcript):
        reason = "No speech was detected in the recording"
        return {
            "criteria": {name: _unassessable(reason) for name in CRITERIA},
            "strengths": [],
            "improvements": [],
            "lesson_summary": "",
        }

    payload = {
        "criteria": CRITERIA,
        "segments": [
            {
                "index": index,
                "start": segment.get("start"),
                "end": segment.get("end"),
                "speaker_role": segment.get("speaker_role", "unknown"),
                "text": segment.get("text", ""),
            }
            for index, segment in enumerate(transcript.get("segments", []))
        ],
    }
    result = generate_json(TEACHING_EFFECTIVENESS_SCORING_INSTRUCTIONS, payload, list_key="criteria")
    entries = _criteria_by_name(result.get("criteria"))
    criteria = {}
    for name in CRITERIA:
        entry = entries.get(name)
        criteria[name] = _normalize_criterion(entry) if entry is not None else _unassessable("Missing model result")
    return {
        **result,
        "criteria": criteria,
        "strengths": _string_list(result.get("strengths")),
        "improvements": _string_list(result.get("improvements")),
        "lesson_summary": str(result.get("lesson_summary") or result.get("summary") or ""),
    }
