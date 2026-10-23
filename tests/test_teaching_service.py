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


_GOOD_OBSERVATIONS = {
    "objective_stated": True,
    "new_concept_taught": True,
    "explains_why": "often",
    "understanding_checks": 3,
    "recap_or_summary": True,
    "homework_or_next_steps": False,
    "feedback_on_student_work": True,
    "teacher_present_until_end": True,
    "first_teaching_minute": 4,
    "instruction_delivered_by_media": False,
    "class_type": "instruction",
}


def _scored(monkeypatch, observations):
    criteria = {name: _criterion(90) for name in CRITERIA}
    return _run(monkeypatch, {"criteria": criteria, "observations": observations})


def test_a_class_meeting_every_observation_keeps_the_model_scores(monkeypatch):
    result = _scored(monkeypatch, _GOOD_OBSERVATIONS)

    assert all(result["criteria"][name]["score"] == 90.0 for name in CRITERIA)
    assert result["observations"]["first_teaching_minute"] == 4.0


def test_missing_behaviour_caps_the_related_criteria(monkeypatch):
    weak = {
        **_GOOD_OBSERVATIONS,
        "objective_stated": False,
        "new_concept_taught": False,
        "recap_or_summary": False,
        "understanding_checks": 0,
        "teacher_present_until_end": False,
        "feedback_on_student_work": False,
    }

    result = _scored(monkeypatch, weak)

    expected = {
        "class_structure": 55, "explanation_clarity": 55, "teaching_methods": 60,
        "lesson_summary": 40, "student_interaction": 55, "pacing": 55, "responsiveness": 60,
    }
    for name, maximum in expected.items():
        assert result["criteria"][name]["score"] == float(maximum)
        assert "limited to" in result["criteria"][name]["score_notes"][0]
    assert result["criteria"]["subject_accuracy"]["score"] == 90.0


def test_a_score_below_the_limit_is_not_raised(monkeypatch):
    criteria = {name: _criterion(30) for name in CRITERIA}

    result = _run(
        monkeypatch,
        {"criteria": criteria, "observations": {**_GOOD_OBSERVATIONS, "objective_stated": False}},
    )

    assert result["criteria"]["class_structure"]["score"] == 30.0
    assert "score_notes" not in result["criteria"]["class_structure"]


def test_zero_checks_is_not_confused_with_false(monkeypatch):
    result = _scored(monkeypatch, {**_GOOD_OBSERVATIONS, "understanding_checks": 0})

    assert result["criteria"]["student_interaction"]["score"] == 55.0
    assert result["criteria"]["class_structure"]["score"] == 90.0


@pytest.mark.parametrize("observations", [None, "bad", [], {}, {"objective_stated": "no", "understanding_checks": "0"}])
def test_missing_or_malformed_observations_apply_no_limits(monkeypatch, observations):
    result = _scored(monkeypatch, observations)

    assert all(result["criteria"][name]["score"] == 90.0 for name in CRITERIA)
