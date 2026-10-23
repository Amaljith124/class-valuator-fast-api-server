"""Named Gemini task instructions used by analysis services."""

SPEAKER_ROLE_INFERENCE_INSTRUCTIONS = """
Identify likely conversational roles from transcript context only.
Allowed roles: teacher, student, unknown.
Do not invent names or claim acoustic speaker diarization.
Preserve every supplied segment's timestamp and text exactly.
Use unknown when evidence is ambiguous or insufficient.
Return JSON with segments [{index, speaker_role, confidence, reason}] and notes.
"""

TRANSCRIPT_TRANSLATION_INSTRUCTIONS = """
Translate each supplied transcript segment into clear, faithful English.
Preserve the meaning, speaker intent, names, and technical terms. Do not summarize,
omit content, add explanations, or change the segment order. Return JSON with
segments [{index, text}] using every supplied index exactly once.
"""

TRANSCRIPT_TRANSLATION_AND_ROLE_INSTRUCTIONS = """
For each transcript segment, return its English text and likely conversational role
using transcript context only. If translation is false, preserve the supplied text
exactly. If translation is true, translate faithfully into English without
summarizing, adding explanations, or changing the order. Allowed roles are teacher,
student, and unknown. Do not invent names or claim acoustic speaker diarization.
Use unknown when evidence is ambiguous or insufficient. Return one JSON object (not a
bare array) of the form {"segments": [{"index": 0, "text": "...", "speaker_role": "teacher",
"confidence": 0.9, "reason": "..."}]}. Return every supplied index exactly once.
Include a short reason only when confidence is below 0.6.
"""

TEACHING_EFFECTIVENESS_SCORING_INSTRUCTIONS = """
Evaluate teaching effectiveness using only the supplied timestamped transcript.
Score each supplied criterion from 0 to 100, or mark it unassessable.
Every score must include exact transcript evidence and a short rationale.
Never fabricate quotes, timestamps, facts, or student responses.
Use the supplied criterion IDs without changing them.
Return one JSON object (not a bare array) with exactly these keys:
- "criteria": an object keyed by each supplied criterion ID, never a list. Each value is
  {"score": <number 0-100 or null>, "assessable": <true or false>,
  "evidence": [{"quote": "<exact words from the transcript>", "start": <seconds>}],
  "rationale": "<one short sentence>", "reason_unassessable": "<only when not assessable>"}
- "strengths": a list of short strings
- "improvements": a list of short strings
- "lesson_summary": a short string
"""

VIDEO_SCREEN_ANALYSIS_INSTRUCTIONS = """
Assess only the supplied sampled classroom or shared-screen frames and timestamped transcript context.
Evaluate readability, visible teaching aids, demonstrations, transitions, and alignment with spoken explanation where evidence permits.
Do not infer unseen frames or continuous events between samples.
Mark missing or unclear evidence as unassessable; a blank screen is not automatically poor.
Return a single JSON object (not an array) of the form {"observations": [ ... ]}, with one entry per sampled frame.
Each observation contains timestamp_seconds, visible_content, readability_score (0..100 or null),
presentation_change, alignment_with_explanation, evidence, and limitations.
"""

EVIDENCE_VALIDATION_REPAIR_INSTRUCTIONS = """
Check the supplied model output against the schema and source evidence.
Do not create new evidence or silently upgrade an unassessable criterion.
If evidence cannot be verified, mark it unassessable and explain why.
Return only the required structured JSON object.
"""
