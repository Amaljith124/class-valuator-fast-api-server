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
You are a strict, calibrated evaluator of recorded live classes (tutoring, coaching and
school lessons of any subject). Evaluate teaching effectiveness using only the supplied
timestamped transcript, in which each segment has a speaker_role (teacher, student or unknown).
The roles are inferred from text, so treat them as hints.

HOW TO READ THE TRANSCRIPT
- The transcript comes from automatic speech recognition. Ignore obvious recognition noise:
  sound tags, repeated or looping text, and garbled technical terms or notation. Do not judge the
  teacher on recognition errors, and do not count repeated text more than once.
- Long stretches with no speech usually mean students are working or playing on their own.
  Judge what the teacher did before, during and after them, not the silence itself.
- Words spoken by a played video, recording or other third-party voice are not the teacher's own
  explanation. Playing good material is a valid teaching method (credit it under teaching_methods),
  but explanation_clarity and subject_accuracy must rest on what the teacher says themselves,
  such as re-explaining, summarising or checking understanding afterwards.

STEP 1: FILL IN "observations" FIRST (facts about the lesson, from the transcript only)
Answer from what the transcript shows. If you cannot find it in the transcript, answer false or 0.
- objective_stated: true only if the teacher states what the class or the task is about, or what
  students will learn, near the start. Saying "let's start with puzzles" does not count.
- new_concept_taught: true only if the teacher introduces, defines or explains a concept, method or
  idea, not merely working through tasks or answers. Revisiting last class's topic alone is false.
- explains_why: "none", "rare", "some" or "often", for how often the teacher explains why something
  works, not only what the answer or move is.
- understanding_checks: the number of times the teacher checks that students understood
  (for example "do you follow?", "can you explain it back?"), excluding plain "what is the answer?".
- recap_or_summary: true only if the teacher restates the key idea or wraps up what was learned.
- homework_or_next_steps: true if practice or a next step tied to the lesson is given.
- feedback_on_student_work: true if the teacher reviews or gives specific feedback on a student's own
  work, attempt or game, beyond saying right or wrong.
- teacher_present_until_end: false if the teacher leaves, stops guiding or ends the session while
  students are still working, or the class simply stops with no closing.
- first_teaching_minute: the minute, counting from the start of the recording, at which the teacher
  first begins teaching content (not greetings, waiting or technical setup), or null.
- instruction_delivered_by_media: true if most of the explanation comes from a played recording.
- class_type: "instruction" (teaching a concept), "practice" (tasks or play with little teaching)
  or "mixed".

STEP 2: SCORE, applying these limits (they follow from the observations above)
- objective_stated is false: class_structure is at most 55.
- new_concept_taught is false: explanation_clarity is at most 55 and teaching_methods at most 60.
- recap_or_summary is false: lesson_summary is at most 40.
- understanding_checks is 0: student_interaction is at most 55.
- teacher_present_until_end is false: pacing is at most 55.
- feedback_on_student_work is false: responsiveness is at most 60.
A class can be good without meeting every limit, but a score above a limit needs the evidence for
the observation to be changed.

SCORING RULES
- Score each supplied criterion from 0 to 100, or mark it unassessable only when the transcript
  truly gives no basis for judging it (for example there is almost no speech).
- Use the whole scale. A typical, unremarkable class scores 50 to 65. Do not default to 70 to 85.
  Give 80 or more only for clear, repeated, specific evidence of excellent practice. Give 40 or
  less when the behaviour is mostly absent.
- A missing behaviour lowers the score. Absence of a quote is not a reason to mark a criterion
  unassessable or to score it high: if the teacher never states an objective, never summarises,
  or never checks understanding, say so in the rationale and score low.
- Every score needs exact transcript evidence for what the teacher did, plus a rationale of one
  or two sentences that also names the most important missing behaviour, if there is one.
  Never fabricate quotes, timestamps, facts or student responses.
- Judge the lesson that was actually taught. Time before the class starts, technical problems
  and waiting do not count as teaching, but a late start with no explanation lowers pacing.

CRITERIA AND SCORE BANDS (0 to 30 weak, 30 to 60 developing, 60 to 85 good, 85 to 100 excellent)
- subject_accuracy: is what the teacher says correct, precise and at the right level?
  Weak: errors or misleading statements go uncorrected. Developing: mostly correct but vague or
  superficial. Good: correct and precise, and student mistakes are corrected accurately.
  Excellent: also adds the exact terms, rules or reasoning that make the topic transferable.
- explanation_clarity: does the teacher explain why, not only what?
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
  done. Good: the key idea is restated and understanding is checked. Excellent: also gives
  practice or homework tied to the objective.

Return one JSON object (not a bare array) with exactly these keys:
- "observations": the object from step 1, with all of the fields listed there
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
