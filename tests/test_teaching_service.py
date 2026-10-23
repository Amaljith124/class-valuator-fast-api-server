import pytest

from app.services import teaching_service
from app.services.teaching_service import CRITERIA


def _transcript(count=3):
    return {
        "segments": [
            {"start": float(index), "end": float(index + 1), "speaker_role": "teacher", "text": f"line {index}"}
            for index in range(count)
        ]
    }


def _criterion(score=80):
    return {"score": score, "assessable": True, "evidence": [{"quote": "line 1"}], "rationale": "ok"}


def _run(monkeypatch, model_result, transcript=None):
    monkeypatch.setattr(teaching_service, "generate_json", lambda *args, **kwargs: model_result)
    return teaching_service.score(transcript or _transcript())


def test_criteria_as_dict_keyed_by_criterion_id(monkeypatch):
    result = _run(monkeypatch, {"criteria": {name: _criterion() for name in CRITERIA}, "strengths": ["a"]})

    assert all(result["criteria"][name]["assessable"] is True for name in CRITERIA)
    assert result["strengths"] == ["a"]


@pytest.mark.parametrize("key", ["id", "criterion_id", "criterion", "name"])
def test_criteria_as_list_of_objects_is_accepted(monkeypatch, key):
    """Regression: Gemini returned `criteria` as a list, which crashed the job with a TypeError."""
    criteria = [{key: name, **_criterion(70)} for name in CRITERIA]

    result = _run(monkeypatch, {"criteria": criteria})

    assert all(result["criteria"][name]["score"] == 70.0 for name in CRITERIA)


def test_criterion_ids_are_normalised(monkeypatch):
    criteria = [{"id": "Subject Accuracy", **_criterion(55)}, {"id": "PACING", **_criterion(65)}]

    result = _run(monkeypatch, {"criteria": criteria})

    assert result["criteria"]["subject_accuracy"]["score"] == 55.0
    assert result["criteria"]["pacing"]["score"] == 65.0
    assert result["criteria"]["class_structure"]["assessable"] is False


def test_missing_or_malformed_entries_become_unassessable_not_errors(monkeypatch):
    criteria = {"subject_accuracy": "very good", "pacing": None, "class_structure": 12}

    result = _run(monkeypatch, {"criteria": criteria})

    assert set(result["criteria"]) == set(CRITERIA)
    assert all(result["criteria"][name]["assessable"] is False for name in CRITERIA)
    assert result["criteria"]["pacing"]["reason_unassessable"] == "Missing model result"


@pytest.mark.parametrize("raw_criteria", [None, "none", 7, [], [1, 2, "x"]])
def test_unusable_criteria_values_never_raise(monkeypatch, raw_criteria):
    result = _run(monkeypatch, {"criteria": raw_criteria})

    assert all(result["criteria"][name]["assessable"] is False for name in CRITERIA)


@pytest.mark.parametrize("raw_score,expected", [("85", 85.0), ("85%", 85.0), (85, 85.0), (84.5, 84.5)])
def test_scores_are_coerced_to_numbers(monkeypatch, raw_score, expected):
    criteria = {name: _criterion(raw_score) for name in CRITERIA}

    result = _run(monkeypatch, {"criteria": criteria})

    assert result["criteria"]["pacing"]["score"] == expected


@pytest.mark.parametrize("raw_score", [True, None, "high", float("nan"), float("inf")])
def test_non_numeric_scores_are_unassessable(monkeypatch, raw_score):
    criteria = {name: _criterion(raw_score) for name in CRITERIA}

    result = _run(monkeypatch, {"criteria": criteria})

    assert result["criteria"]["pacing"]["assessable"] is False
    assert result["criteria"]["pacing"]["score"] is None


def test_assessable_flag_defaults_from_score_when_omitted(monkeypatch):
    criteria = {name: {"score": 60, "evidence": ["line 1"]} for name in CRITERIA}

    result = _run(monkeypatch, {"criteria": criteria})

    assert result["criteria"]["pacing"]["assessable"] is True


def test_strengths_improvements_and_summary_are_normalised(monkeypatch):
    result = _run(
        monkeypatch,
        {
            "criteria": {},
            "strengths": "Clear examples",
            "improvements": [{"text": "More pauses"}, "Check understanding", "", None],
            "lesson_summary": None,
        },
    )

    assert result["strengths"] == ["Clear examples"]
    assert result["improvements"] == ["More pauses", "Check understanding"]
    assert result["lesson_summary"] == ""


def test_empty_transcript_skips_the_model_call(monkeypatch):
    def must_not_be_called(*args, **kwargs):
        raise AssertionError("Gemini must not be called for a transcript without speech")

    monkeypatch.setattr(teaching_service, "generate_json", must_not_be_called)

    result = teaching_service.score({"segments": []})

    assert all(result["criteria"][name]["assessable"] is False for name in CRITERIA)
    assert "speech" in result["criteria"]["pacing"]["reason_unassessable"].lower()


def test_blank_segments_count_as_no_speech(monkeypatch):
    monkeypatch.setattr(teaching_service, "generate_json", lambda *a, **k: pytest.fail("called"))

    result = teaching_service.score({"segments": [{"text": "  "}, {"text": ""}]})

    assert all(result["criteria"][name]["assessable"] is False for name in CRITERIA)
