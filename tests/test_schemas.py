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
