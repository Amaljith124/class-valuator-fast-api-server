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


@pytest.mark.parametrize("bad_score", [101, -1, 1000])
def test_out_of_range_score_marks_only_that_criterion_unassessable(bad_score):
    teaching = _teaching_result(
        {"score": bad_score, "evidence": ["present"], "assessable": True}
    )

    result = combine(teaching, {}, {}, {"text": "present"})

    criterion = result["criteria"]["subject_accuracy"]
    assert criterion["assessable"] is False
    assert criterion["reason_unassessable"] == "Score outside allowed range"


def test_quote_comparison_ignores_punctuation():
    teaching = _teaching_result(
        {"score": 70, "evidence": ["so, today we learn: fractions!"], "assessable": True}
    )

    result = combine(teaching, {}, {}, {"text": "So today we learn fractions"})

    assert result["criteria"]["subject_accuracy"]["assessable"] is True


def test_quotes_may_skip_words_with_an_ellipsis():
    teaching = _teaching_result(
        {"score": 70, "evidence": [{"quote": "open your books ... page twelve"}], "assessable": True}
    )

    result = combine(teaching, {}, {}, {"text": "Please open your books and turn to page twelve."})

    assert result["criteria"]["subject_accuracy"]["assessable"] is True


def test_every_fragment_of_an_ellipsis_quote_must_exist():
    teaching = _teaching_result(
        {"score": 70, "evidence": ["open your books ... page ninety"], "assessable": True}
    )

    result = combine(teaching, {}, {}, {"text": "Please open your books and turn to page twelve."})

    assert result["criteria"]["subject_accuracy"]["assessable"] is False


@pytest.mark.parametrize(
    "evidence",
    [[], None, "a string", [123], [{"quote": 5}], [{}]],
)
def test_missing_or_unusable_evidence_is_unassessable_not_an_error(evidence):
    teaching = _teaching_result({"score": 70, "evidence": evidence, "assessable": True})

    result = combine(teaching, {}, {}, {"text": "Some transcript."})

    criterion = result["criteria"]["subject_accuracy"]
    assert criterion["assessable"] is False
    assert criterion["score"] is None


def test_evidence_objects_may_use_a_text_key():
    teaching = _teaching_result(
        {"score": 70, "evidence": [{"text": "the class started"}], "assessable": True}
    )

    result = combine(teaching, {}, {}, {"text": "The class started today."})

    assert result["criteria"]["subject_accuracy"]["assessable"] is True


@pytest.mark.parametrize("criteria", [None, [], "x", {"subject_accuracy": "bad"}])
def test_malformed_teaching_payload_never_raises(criteria):
    result = combine({"criteria": criteria}, {}, {}, {"text": "text"})

    assert result["overall_score"] is None
    assert all(item["assessable"] is False for item in result["criteria"].values())


def test_overall_score_is_a_weighted_mean_over_assessable_criteria_only():
    teaching = _teaching_result({"score": 80, "evidence": ["a"], "assessable": True})
    teaching["criteria"]["pacing"] = {"score": 60, "evidence": ["b"], "assessable": True}

    result = combine(teaching, {}, {}, {"text": "a b"})

    from app.services.evidence_service import WEIGHTS

    weight_a, weight_b = WEIGHTS["subject_accuracy"], WEIGHTS["pacing"]
    assert result["overall_score"] == pytest.approx(
        (80 * weight_a + 60 * weight_b) / (weight_a + weight_b), abs=0.01
    )
    assert result["score_coverage"] == pytest.approx(weight_a + weight_b)


def test_score_notes_and_observations_reach_the_report():
    teaching = _teaching_result(
        {"score": 40, "evidence": ["present"], "assessable": True, "score_notes": ["Score limited to 40"]}
    )
    teaching["observations"] = {"objective_stated": False}

    result = combine(teaching, {}, {}, {"text": "present"})

    assert result["criteria"]["subject_accuracy"]["score_notes"] == ["Score limited to 40"]
    assert result["lesson_observations"] == {"objective_stated": False}


def test_weights_cover_every_criterion_and_sum_to_one():
    from app.services.evidence_service import WEIGHTS

    assert set(WEIGHTS) == set(CRITERIA)
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
