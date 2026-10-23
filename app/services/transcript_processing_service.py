import asyncio

from app.core.config import settings
from app.services.gemini_service import generate_json
from app.templates.prompt_templates import TRANSCRIPT_TRANSLATION_AND_ROLE_INSTRUCTIONS

_llm_semaphore = asyncio.Semaphore(settings.llm_concurrency)
TRANSLATION_MODEL = "gemini-3.5-flash-lite"


async def translate_and_label(transcript: dict) -> tuple[dict, dict, bool, dict]:
    source_language = str(transcript.get("language") or "").strip().lower()
    needs_translation = not (source_language == "en" or source_language.startswith("en-"))
    source_segments = transcript.get("segments", [])
    batch_size = settings.llm_batch_size
    batches = [
        (start, source_segments[start:start + batch_size])
        for start in range(0, len(source_segments), batch_size)
    ]

    async def process_batch(start: int, batch: list[dict]) -> tuple[int, list[dict]]:
        payload_segments = [
            {
                "index": start + offset,
                "start": segment.get("start"),
                "text": str(segment.get("text") or ""),
            }
            for offset, segment in enumerate(batch)
        ]
        async with _llm_semaphore:
            response = await asyncio.to_thread(
                generate_json,
                TRANSCRIPT_TRANSLATION_AND_ROLE_INSTRUCTIONS,
                {
                    "source_language": source_language or "unknown",
                    "target_language": "English",
                    "translate": needs_translation,
                    "segments": payload_segments,
                },
                False,
                TRANSLATION_MODEL,
            )
        result_segments = response.get("segments")
        if not isinstance(result_segments, list):
            raise ValueError("Gemini transcript processing did not return a segment list")
        return start, result_segments

    results = await asyncio.gather(
        *(process_batch(start, batch) for start, batch in batches),
        return_exceptions=True,
    )
    labels = {}
    for (start, batch), batch_result in zip(batches, results):
        if isinstance(batch_result, BaseException):
            raise batch_result
        _, result_segments = batch_result
        expected_indexes = set(range(start, start + len(batch)))
        for result_segment in result_segments:
            if not isinstance(result_segment, dict):
                continue
            index = result_segment.get("index")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index not in expected_indexes
                or index in labels
            ):
                raise ValueError("Gemini transcript processing returned an invalid or duplicate index")
            role = result_segment.get("speaker_role")
            if role not in {"teacher", "student", "unknown"}:
                raise ValueError("Gemini transcript processing returned an invalid speaker role")
            confidence = result_segment.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                confidence = 0.0
            text = result_segment.get("text")
            if not isinstance(text, str):
                raise ValueError("Gemini transcript processing returned invalid segment text")
            labels[index] = {
                "text": text if needs_translation else str(source_segments[index].get("text") or ""),
                "speaker_role": role,
                "role_confidence": max(0.0, min(1.0, float(confidence))),
                "role_evidence": str(result_segment.get("reason") or "")[:160]
                if confidence < 0.6 else "",
            }

    if len(labels) != len(source_segments) or set(labels) != set(range(len(source_segments))):
        raise ValueError("Gemini transcript processing omitted one or more segment indexes")

    source_output = {**transcript, "segments": []}
    english_output = {
        **transcript,
        "language": "en",
        "source_language": transcript.get("language"),
        "segments": [],
    }
    for index, source_segment in enumerate(source_segments):
        label = labels[index]
        source_output["segments"].append(
            {
                **source_segment,
                "speaker_role": label["speaker_role"],
                "role_confidence": label["role_confidence"],
                "role_evidence": label["role_evidence"],
            }
        )
        english_output["segments"].append(
            {
                **source_segment,
                "text": label["text"],
                "speaker_role": label["speaker_role"],
                "role_confidence": label["role_confidence"],
                "role_evidence": label["role_evidence"],
            }
        )

    english_output["text"] = " ".join(
        segment["text"] for segment in english_output["segments"] if segment["text"]
    )
    source_output["text"] = transcript.get("text", "")
    roles = {
        "segments": english_output["segments"],
        "limitations": [
            "Roles are inferred from transcript context; this is not acoustic diarization."
        ],
    }
    return source_output, english_output, needs_translation, roles
