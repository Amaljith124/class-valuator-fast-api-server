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

LESSON_OBSERVATION_INSTRUCTIONS = """
You record facts about a recorded live class (tutoring, coaching or school lesson of any subject)
from its timestamped transcript. Do not score or judge quality; only report what happened.
Each segment has a speaker_role (teacher, student or unknown); the roles are inferred from text,
so treat them as hints. The transcript comes from automatic speech recognition: ignore sound tags,
repeated text and garbled terms.

Answer from what the transcript shows. If you cannot find it in the transcript, answer false or 0.
- objective_stated: true only if the teacher states what the class or the task is about, or what
  students will learn, near the start. Saying "let's start with puzzles" does not count.
- new_concept_taught: true only if the teacher (or material the teacher plays) introduces, defines
  or explains a concept, method or idea, not merely working through tasks or answers. Revisiting
  last class's topic alone is false.
- explains_why: "none", "rare", "some" or "often", for how often the teacher explains why something
  works, not only what the answer or move is.
- understanding_checks: the number of times the teacher checks that students understood
  (for example "do you follow?", "can you explain it back?"), excluding plain "what is the answer?".
- recap_or_summary: true if the key idea is restated or what was learned is wrapped up. A recap
  given by a played recording counts when the teacher uses it, for example by referring back to it
  or checking afterwards that students understood it. A class that simply stops does not count.
- homework_or_next_steps: true if practice or a next step tied to the lesson is given.
- feedback_on_student_work: true if the teacher reviews or corrects a student's own work, attempt or
  game with a specific comment, beyond saying right or wrong.
- teacher_present_until_end: false if the teacher leaves, stops guiding or ends the session while
  students are still working, or the class simply stops with no closing.
- first_teaching_minute: the minute, counting from the start of the recording, at which the teacher
  first begins teaching content (not greetings, waiting or technical setup), or null.
- instruction_delivered_by_media: true if a large part of the explanation comes from a played
  recording (the voice of someone who is not the class teacher).
- media_without_follow_up: true only if a played recording supplied the explanation and the teacher
  did nothing with it: no framing before it, and no re-explaining, questions or checks afterwards.
  If the teacher introduces, pauses, re-explains or checks understanding, this is false.
- class_type: "instruction" (teaching a concept), "practice" (tasks or play with little teaching)
  or "mixed".
Return one JSON object with exactly these fields.
"""

TEACHING_EFFECTIVENESS_SCORING_INSTRUCTIONS = """
You are a strict, calibrated evaluator of recorded live classes (tutoring, coaching and
school lessons of any subject). Evaluate teaching effectiveness using the supplied timestamped
transcript, in which each segment has a speaker_role (teacher, student or unknown); the roles are
inferred from text, so treat them as hints. The payload also contains "observations": established
facts about the lesson. Use them as given and do not contradict them.

HOW TO READ THE TRANSCRIPT
- The transcript comes from automatic speech recognition. Ignore obvious recognition noise:
  sound tags, repeated or looping text, and garbled technical terms or notation. Do not judge the
  teacher on recognition errors, and do not count repeated text more than once.
- Long stretches with no speech usually mean students are working or playing on their own.
  Judge what the teacher did before, during and after them, not the silence itself.
- Choosing, introducing and following up a good recorded explanation is a valid way to teach.
  When a teacher plays material, frames it, re-explains the key point or checks understanding, treat
  the whole sequence as the teacher's instruction and judge its quality. Only when material is played
  with no framing or follow-up (see media_without_follow_up) should the explanation not be credited.

SCORING RULES
- Score each supplied criterion from 0 to 100, or mark it unassessable only when the transcript
  truly gives no basis for judging it (for example there is almost no speech).
- Use the whole scale. A typical, unremarkable class scores 50 to 65. Do not default to 70 to 85.
  Give 80 or more only for clear, repeated, specific evidence of excellent practice. Give 40 or
  less when the behaviour is mostly absent.
- A missing behaviour lowers the score. Absence of a quote is not a reason to mark a criterion
  unassessable or to score it high: if the teacher never states an objective, never summarises,
  or never checks understanding, say so in the rationale and score low.
- Every score needs exact transcript evidence for what the teacher did, chosen to show the
  behaviour being scored (not a greeting or filler), plus a rationale of one or two sentences that
  also names the most important missing behaviour, if there is one.
  Never fabricate quotes, timestamps, facts or student responses.
- Judge the lesson that was actually taught. Time before the class starts, technical problems
  and waiting do not count as teaching, but a late start with no explanation lowers pacing.
- Score limits that follow from the observations (the system also enforces them):
  objective_stated false: class_structure at most 55. new_concept_taught false: explanation_clarity
  at most 55 and teaching_methods at most 60. recap_or_summary false: lesson_summary at most 40.
  understanding_checks 0: student_interaction at most 55. teacher_present_until_end false: pacing at
  most 55. feedback_on_student_work false: responsiveness at most 60. media_without_follow_up true:
  explanation_clarity at most 55.

CRITERIA AND SCORE BANDS (0 to 30 weak, 30 to 60 developing, 60 to 85 good, 85 to 100 excellent)
- subject_accuracy: is what is taught correct, precise and at the right level?
  Weak: errors or misleading statements go uncorrected. Developing: mostly correct but vague or
  superficial. Good: correct and precise, and student mistakes are corrected accurately.
  Excellent: also adds the exact terms, rules or reasoning that make the topic transferable.
- explanation_clarity: does the instruction explain why, not only what?
  Weak: answers or moves are given with no reasoning. Developing: some reasoning, but unclear or
  rushed. Good: concepts are defined, shown with an example and linked to what students know.
  Excellent: also re-explains in a second way when students are unsure.
- teaching_methods: variety and fit of methods (worked examples, guided questions, demonstration,
  practice that matches the concept, use of media, feedback on student work).
  Weak: only one routine, such as asking for answers with nothing taught first. Developing: a
  method is used but not matched to a concept. Good: a deliberate sequence, for example concept,
  example, guided practice. Excellent: also adapts the method to the students' responses.
- class_structure: objective, order and close.
  Weak: no stated topic or objective, jumps straight into tasks, no wrap-up. Developing: a topic
  is named, but the order is loose or the ending is abrupt. Good: stated objective, logical
  sequence, a recap or closing statement. Excellent: also links to prior learning and next steps
  or practice to do before the next class.
- pacing: use of time and rhythm.
  Weak: long late start, long unproductive waiting, or the teacher leaves while students are
  still working. Developing: uneven pace, with some stretches without teaching or guidance.
  Good: steady pace and enough thinking time for students to answer. Excellent: also adjusts the
  pace to student difficulty.
- student_interaction: quality, not quantity, of student involvement. Many short questions of the
  kind "what do we play here?" do not count as high interaction unless they follow an explanation
  and probe understanding.
  Weak: the teacher mostly talks, or only fires questions for quick answers. Developing: students
  answer and sometimes ask, with little follow-up. Good: questions that check understanding,
  named students invited, student questions answered. Excellent: students explain their reasoning
  and the teacher builds on it.
- responsiveness: reaction to what students say and struggle with.
  Weak: wrong or missing answers are ignored or only corrected with the answer. Developing:
  corrections are given, but not why. Good: errors are diagnosed and explained, and individual
  students (including new ones) get attention. Excellent: also gives specific feedback on a
  student's own work, such as reviewing their game, essay or solution.
- lesson_summary: recap of what was learned.
  Weak: no recap and no closing, or the class just ends. Developing: a brief mention of what was
  done. Good: the key idea is restated, by the teacher or by material the teacher plays and then
  follows up, and understanding is checked. Excellent: also gives practice or homework tied to the
  objective.

Return one JSON object (not a bare array) with exactly these keys:
- "criteria": an object keyed by each supplied criterion ID, never a list. Each value is
  {"score": <number 0-100 or null>, "assessable": <true or false>,
  "evidence": [{"quote": "<exact words from the transcript>", "start": <seconds>}],
  "rationale": "<one or two sentences, naming the main missing behaviour if any>",
  "reason_unassessable": "<only when not assessable>"}
- "strengths": a list of short strings
- "improvements": a list of short, specific strings, most important first
- "lesson_summary": a short string
Use the supplied criterion IDs without changing them.
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
