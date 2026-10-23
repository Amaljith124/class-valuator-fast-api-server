import pytest

from app.services.evidence_service import combine
from app.services.teaching_service import CRITERIA


def _teaching_result(subject_accuracy):
    criteria = {
        name: {"score": None, "evidence": [], "assessable": False}
        for name in CRITERIA
    }
    criteria["subject_accuracy"] = subject_accuracy
    return {"criteria": criteria}


def test_unverified_quote_marks_only_that_criterion_unassessable():
    teaching = _teaching_result(
        {"score": 82, "evidence": ["a quote not in the transcript"], "assessable": True}
    )

    result = combine(teaching, {}, {}, {"text": "Transcript content."})

    criterion = result["criteria"]["subject_accuracy"]
    assert criterion["assessable"] is False
    assert criterion["score"] is None
    assert criterion["reason_unassessable"] == "Evidence quote could not be verified in transcript"


def test_quote_validation_normalizes_case_and_whitespace():
    teaching = _teaching_result(
        {"score": 82, "evidence": ["THE   CLASS\n started"], "assessable": True}
    )

    result = combine(teaching, {}, {}, {"text": "The class started today."})

    assert result["criteria"]["subject_accuracy"]["assessable"] is True
    assert result["criteria"]["subject_accuracy"]["score"] == 82.0


def test_out_of_range_score_still_raises():
    teaching = _teaching_result(
        {"score": 101, "evidence": ["present"], "assessable": True}
    )

    with pytest.raises(ValueError, match="outside allowed range"):
        combine(teaching, {}, {}, {"text": "present"})
