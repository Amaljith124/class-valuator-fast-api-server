from app.templates.prompt_templates import (
    SPEAKER_ROLE_INFERENCE_INSTRUCTIONS,
    TEACHING_EFFECTIVENESS_SCORING_INSTRUCTIONS,
    VIDEO_SCREEN_ANALYSIS_INSTRUCTIONS,
)


def test_required_prompt_instructions_are_present_and_non_empty() -> None:
    required_instructions = (
        SPEAKER_ROLE_INFERENCE_INSTRUCTIONS,
        TEACHING_EFFECTIVENESS_SCORING_INSTRUCTIONS,
        VIDEO_SCREEN_ANALYSIS_INSTRUCTIONS,
    )
    assert all(instructions.strip() for instructions in required_instructions)


def test_prompts_preserve_unknown_or_unassessable_when_evidence_is_insufficient() -> None:
    assert "unknown" in SPEAKER_ROLE_INFERENCE_INSTRUCTIONS.lower()
    assert "unassessable" in TEACHING_EFFECTIVENESS_SCORING_INSTRUCTIONS.lower()
    assert "unassessable" in VIDEO_SCREEN_ANALYSIS_INSTRUCTIONS.lower()
    assert "never fabricate" in TEACHING_EFFECTIVENESS_SCORING_INSTRUCTIONS.lower()


def test_scoring_prompt_defines_a_rubric_for_every_criterion() -> None:
    from app.services.teaching_service import CRITERIA

    for name in CRITERIA:
        assert f"- {name}:" in TEACHING_EFFECTIVENESS_SCORING_INSTRUCTIONS
