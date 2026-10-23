from app.services.gemini_service import generate_json
from app.templates.prompt_templates import SPEAKER_ROLE_INFERENCE_INSTRUCTIONS

def infer_roles(transcript: dict) -> dict:
    segments = transcript.get("segments", [])
    payload_segments = [
        {"index": index, "start": segment.get("start"), "text": segment.get("text", "")}
        for index, segment in enumerate(segments)
    ]
    result = generate_json(
        SPEAKER_ROLE_INFERENCE_INSTRUCTIONS,
        {"segments": payload_segments},
        False,
    )
    labels = {x.get("index"):x for x in result.get("segments",[]) if isinstance(x,dict)}
    for i, item in enumerate(segments):
        label = labels.get(i,{})
        role = label.get("speaker_role","unknown")
        item["speaker_role"] = role if role in ("teacher","student","unknown") else "unknown"
        item["role_confidence"] = label.get("confidence",0.0)
        item["role_evidence"] = str(label.get("reason",""))[:300]
    return {"segments":segments,"limitations":["Roles are inferred from transcript context; this is not acoustic diarization."]}
