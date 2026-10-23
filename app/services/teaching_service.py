from app.services.gemini_service import generate_json
from app.templates.prompt_templates import TEACHING_EFFECTIVENESS_SCORING_INSTRUCTIONS
CRITERIA = ["subject_accuracy","explanation_clarity","teaching_methods","class_structure","pacing","student_interaction","responsiveness","lesson_summary"]

def score(transcript: dict) -> dict:
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
    result = generate_json(TEACHING_EFFECTIVENESS_SCORING_INSTRUCTIONS,payload)
    criteria = result.get("criteria",{})
    for name in CRITERIA:
        if name not in criteria or not isinstance(criteria[name],dict):
            criteria[name] = {"score":None,"evidence":[],"assessable":False,"reason_unassessable":"Missing model result"}
    result["criteria"] = criteria
    return result
