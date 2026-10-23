import pytest

from app.services import schemas
from app.services.evidence_service import WEIGHTS
from app.services.teaching_service import CRITERIA

jsonschema = pytest.importorskip("jsonschema")


def test_criteria_names_match_the_scoring_code():
    assert schemas.CRITERIA_NAMES == CRITERIA
    assert set(schemas.CRITERIA_NAMES) == set(WEIGHTS)


@pytest.mark.parametrize("name", ["TEACHING_SCHEMA", "SEGMENTS_SCHEMA", "VIDEO_SCHEMA"])
def test_schemas_are_valid_json_schema(name):
    jsonschema.Draft202012Validator.check_schema(getattr(schemas, name))


def test_teaching_schema_requires_every_criterion():
    criterion = {"score": None, "assessable": False, "evidence": []}
    good = {
        "criteria": {name: criterion for name in CRITERIA},
        "strengths": [],
        "improvements": [],
        "lesson_summary": "",
    }
    jsonschema.validate(good, schemas.TEACHING_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**good, "criteria": [dict(criterion, id=name) for name in CRITERIA]}, schemas.TEACHING_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**good, "criteria": {}}, schemas.TEACHING_SCHEMA)


def test_segments_schema_limits_roles():
    segment = {"index": 0, "text": "hi", "speaker_role": "teacher", "confidence": 0.9}
    jsonschema.validate({"segments": [segment]}, schemas.SEGMENTS_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"segments": [{**segment, "speaker_role": "principal"}]}, schemas.SEGMENTS_SCHEMA)


def test_observations_schema_requires_every_fact():
    observations = {
        "objective_stated": True, "new_concept_taught": True, "explains_why": "some",
        "understanding_checks": 1, "recap_or_summary": False, "homework_or_next_steps": False,
        "feedback_on_student_work": False, "teacher_present_until_end": True,
        "first_teaching_minute": None, "instruction_delivered_by_media": False,
        "media_without_follow_up": False, "class_type": "mixed",
    }
    jsonschema.validate(observations, schemas.OBSERVATIONS_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({key: value for key, value in observations.items() if key != "media_without_follow_up"}, schemas.OBSERVATIONS_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**observations, "class_type": "lecture"}, schemas.OBSERVATIONS_SCHEMA)
