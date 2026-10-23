import json
import logging
import math
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from app.services.gemini_service import generate_json
from app.services.schemas import OBSERVATIONS_SCHEMA, TEACHING_SCHEMA
from app.templates.prompt_templates import (
    LESSON_OBSERVATION_INSTRUCTIONS,
    TEACHING_EFFECTIVENESS_SCORING_INSTRUCTIONS,
)

logger = logging.getLogger(__name__)
# Facts are asked for several times with some randomness and the majority answer is kept, so one
# unlucky answer cannot swing a class's score. The scoring call itself is pinned with a seed.
OBSERVATION_VOTES = 3
OBSERVATION_TEMPERATURE = 0.4
SCORING_SEED = 7

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


# (observation, value that triggers the limit, criterion, maximum score, explanation)
SCORE_LIMITS = [
    ("objective_stated", False, "class_structure", 55, "No learning objective was stated"),
    ("new_concept_taught", False, "explanation_clarity", 55, "No new concept was taught or explained"),
    ("new_concept_taught", False, "teaching_methods", 60, "No new concept was taught or explained"),
    ("recap_or_summary", False, "lesson_summary", 40, "No recap or summary was given"),
    ("understanding_checks", 0, "student_interaction", 55, "No checks for understanding were made"),
    ("teacher_present_until_end", False, "pacing", 55, "Teacher did not guide the class through to its end"),
    ("feedback_on_student_work", False, "responsiveness", 60, "No specific feedback on student work"),
    ("media_without_follow_up", True, "explanation_clarity", 55, "A played recording was not framed or followed up"),
]
# Flags where true is the worse answer, so a tied vote leans to true.
_PROBLEM_FLAGS = {"media_without_follow_up"}
_OBSERVATION_FLAGS = (
    "objective_stated", "new_concept_taught", "recap_or_summary", "homework_or_next_steps",
    "feedback_on_student_work", "teacher_present_until_end", "instruction_delivered_by_media",
    "media_without_follow_up",
)


def _normalize_observations(raw) -> dict:
    """Keep only well-typed observations; anything missing stays out so no limit is applied for it."""
    if not isinstance(raw, dict):
        return {}
    observations: dict = {}
    for name in _OBSERVATION_FLAGS:
        if isinstance(raw.get(name), bool):
            observations[name] = raw[name]
    checks = raw.get("understanding_checks")
    if isinstance(checks, (int, float)) and not isinstance(checks, bool) and math.isfinite(checks) and checks >= 0:
        observations["understanding_checks"] = int(checks)
    if raw.get("explains_why") in {"none", "rare", "some", "often"}:
        observations["explains_why"] = raw["explains_why"]
    if raw.get("class_type") in {"instruction", "practice", "mixed"}:
        observations["class_type"] = raw["class_type"]
    minute = _coerce_score(raw.get("first_teaching_minute"))
    observations["first_teaching_minute"] = minute if minute is not None and minute >= 0 else None
    return observations


def _merge_observations(votes: list[dict]) -> dict:
    """Majority answer per field across the normalised votes; ties favour the more cautious answer."""
    merged: dict = {}
    for name in _OBSERVATION_FLAGS:
        answers = [vote[name] for vote in votes if name in vote]
        if answers:
            yes = sum(answers)
            if yes * 2 == len(answers):
                # A tie: assume the behaviour was absent, except for flags that describe a problem.
                merged[name] = name in _PROBLEM_FLAGS
            else:
                merged[name] = yes * 2 > len(answers)
    checks = [vote["understanding_checks"] for vote in votes if "understanding_checks" in vote]
    if checks:
        merged["understanding_checks"] = int(statistics.median_low(checks))
    for name in ("explains_why", "class_type"):
        answers = [vote[name] for vote in votes if name in vote]
        if answers:
            merged[name] = Counter(answers).most_common(1)[0][0]
    minutes = [vote["first_teaching_minute"] for vote in votes if vote.get("first_teaching_minute") is not None]
    merged["first_teaching_minute"] = round(float(statistics.median(minutes)), 2) if minutes else None
    return merged


def _observe(segments_payload: list[dict]) -> dict:
    """Ask for the lesson facts several times and keep the majority answer."""

    def ask(seed: int):
        return generate_json(
            LESSON_OBSERVATION_INSTRUCTIONS,
            {"segments": segments_payload},
            thinking=False,
            schema=OBSERVATIONS_SCHEMA,
            temperature=OBSERVATION_TEMPERATURE,
            seed=seed,
        )

    with ThreadPoolExecutor(max_workers=OBSERVATION_VOTES) as executor:
        futures = [executor.submit(ask, seed) for seed in range(1, OBSERVATION_VOTES + 1)]
    votes = []
    for future in futures:
        try:
            votes.append(_normalize_observations(future.result()))
        except Exception as error:
            logger.warning("[teaching.observe] one observation answer failed and is ignored: %s", error)
    if not votes:
        logger.warning("[teaching.observe] no observation answer was usable; no score limits will apply")
        return {}
    merged = _merge_observations(votes)
    merged["votes"] = len(votes)
    return merged


def _apply_score_limits(criteria: dict[str, dict], observations: dict) -> None:
    for observation, trigger, name, maximum, reason in SCORE_LIMITS:
        value = observations.get(observation)
        # bool is an int subclass, so compare types as well to keep False apart from 0.
        if observation not in observations or type(value) is not type(trigger) or value != trigger:
            continue
        criterion = criteria.get(name)
        if criterion and criterion["assessable"] and criterion["score"] is not None and criterion["score"] > maximum:
            criterion["score_notes"] = criterion.get("score_notes", []) + [
                f"Score limited to {maximum} (model gave {criterion['score']:g}): {reason}"
            ]
            criterion["score"] = float(maximum)


def _has_speech(transcript: dict) -> bool:
    return any(str(segment.get("text") or "").strip() for segment in transcript.get("segments", []))


def score(transcript: dict) -> dict:
    if not _has_speech(transcript):
        reason = "No speech was detected in the recording"
        return {
            "criteria": {name: _unassessable(reason) for name in CRITERIA},
            "observations": {},
            "strengths": [],
            "improvements": [],
            "lesson_summary": "",
        }

    segments_payload = [
        {
            "index": index,
            "start": segment.get("start"),
            "end": segment.get("end"),
            "speaker_role": segment.get("speaker_role", "unknown"),
            "text": segment.get("text", ""),
        }
        for index, segment in enumerate(transcript.get("segments", []))
    ]
    observations = _observe(segments_payload)
    payload = {"criteria": CRITERIA, "observations": observations, "segments": segments_payload}
    result = generate_json(
        TEACHING_EFFECTIVENESS_SCORING_INSTRUCTIONS,
        payload,
        list_key="criteria",
        schema=TEACHING_SCHEMA,
        seed=SCORING_SEED,
    )
    entries = _criteria_by_name(result.get("criteria"))
    criteria = {}
    for name in CRITERIA:
        entry = entries.get(name)
        criteria[name] = _normalize_criterion(entry) if entry is not None else _unassessable("Missing model result")
    _apply_score_limits(criteria, observations)
    return {
        **result,
        "observations": observations,
        "criteria": criteria,
        "strengths": _string_list(result.get("strengths")),
        "improvements": _string_list(result.get("improvements")),
        "lesson_summary": str(result.get("lesson_summary") or result.get("summary") or ""),
    }
