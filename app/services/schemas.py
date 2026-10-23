"""JSON Schemas passed to Gemini so the model must return the exact shape we parse."""

CRITERIA_NAMES = [
    "subject_accuracy",
    "explanation_clarity",
    "teaching_methods",
    "class_structure",
    "pacing",
    "student_interaction",
    "responsiveness",
    "lesson_summary",
]

_NULLABLE_NUMBER = {"type": ["number", "null"]}
_NULLABLE_STRING = {"type": ["string", "null"]}


def _object(properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
    }


_CRITERION = _object(
    {
        "score": {"type": ["number", "null"], "description": "0 to 100, or null when not assessable"},
        "assessable": {"type": "boolean"},
        "evidence": {
            "type": "array",
            "items": _object({"quote": {"type": "string"}, "start": _NULLABLE_NUMBER}, ["quote"]),
        },
        "rationale": {"type": "string"},
        "reason_unassessable": _NULLABLE_STRING,
    },
    ["score", "assessable", "evidence"],
)

_OBSERVATIONS = _object(
    {
        "objective_stated": {"type": "boolean"},
        "new_concept_taught": {"type": "boolean"},
        "explains_why": {"type": "string", "enum": ["none", "rare", "some", "often"]},
        "understanding_checks": {"type": "integer"},
        "recap_or_summary": {"type": "boolean"},
        "homework_or_next_steps": {"type": "boolean"},
        "feedback_on_student_work": {"type": "boolean"},
        "teacher_present_until_end": {"type": "boolean"},
        "first_teaching_minute": _NULLABLE_NUMBER,
        "instruction_delivered_by_media": {"type": "boolean"},
        "class_type": {"type": "string", "enum": ["instruction", "practice", "mixed"]},
    }
)

TEACHING_SCHEMA = _object(
    {
        "observations": _OBSERVATIONS,
        "criteria": _object({name: _CRITERION for name in CRITERIA_NAMES}),
        "strengths": {"type": "array", "items": {"type": "string"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
        "lesson_summary": {"type": "string"},
    }
)

SEGMENTS_SCHEMA = _object(
    {
        "segments": {
            "type": "array",
            "items": _object(
                {
                    "index": {"type": "integer"},
                    "text": {"type": "string"},
                    "speaker_role": {"type": "string", "enum": ["teacher", "student", "unknown"]},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                ["index", "text", "speaker_role", "confidence"],
            ),
        }
    }
)

VIDEO_SCHEMA = _object(
    {
        "observations": {
            "type": "array",
            "items": _object(
                {
                    "timestamp_seconds": _NULLABLE_NUMBER,
                    "visible_content": {"type": "string"},
                    "readability_score": _NULLABLE_NUMBER,
                    "presentation_change": {"type": "string"},
                    "alignment_with_explanation": {"type": "string"},
                    "evidence": {"type": "string"},
                    "limitations": {"type": "string"},
                },
                ["timestamp_seconds", "visible_content", "readability_score"],
            ),
        }
    }
)
