import asyncio
import logging
import math

from app.core.config import settings
from app.services.gemini_service import generate_json
from app.services.schemas import SEGMENTS_SCHEMA
from app.templates.prompt_templates import TRANSCRIPT_TRANSLATION_AND_ROLE_INSTRUCTIONS

logger = logging.getLogger(__name__)
_llm_semaphore = asyncio.Semaphore(settings.llm_concurrency)
VALID_ROLES = {"teacher", "student", "unknown"}
BATCH_ATTEMPTS = 2
# More than this share of segments missing means the model output is not trustworthy.
MAX_FALLBACK_SHARE = 0.2
MIN_FALLBACK_ALLOWANCE = 3


def _clean_role(value) -> str:
    role = value.strip().lower() if isinstance(value, str) else ""
    return role if role in VALID_ROLES else "unknown"


def _clean_confidence(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _clean_index(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _parse_batch(result_segments, expected_indexes: range, needs_translation: bool) -> dict[int, dict]:
    """Keep every usable entry; ignore invalid, out-of-range and duplicate ones."""
    parsed: dict[int, dict] = {}
    if not isinstance(result_segments, list):
        return parsed
    for result_segment in result_segments:
        if not isinstance(result_segment, dict):
            continue
        index = _clean_index(result_segment.get("index"))
        if index is None or index not in expected_indexes or index in parsed:
            continue
        text = result_segment.get("text")
        if needs_translation and not (isinstance(text, str) and text.strip()):
            continue
        confidence = _clean_confidence(result_segment.get("confidence"))
        parsed[index] = {
            "text": text if isinstance(text, str) else "",
            "speaker_role": _clean_role(result_segment.get("speaker_role")),
            "role_confidence": confidence,
            "role_evidence": str(result_segment.get("reason") or "")[:160] if confidence < 0.6 else "",
        }
    return parsed


async def translate_and_label(transcript: dict) -> tuple[dict, dict, bool, dict]:
    source_language = str(transcript.get("language") or "").strip().lower()
    needs_translation = not (source_language == "en" or source_language.startswith("en-"))
    source_segments = transcript.get("segments", [])
    batch_size = max(1, settings.llm_batch_size)
    batches = [
        (start, source_segments[start:start + batch_size])
        for start in range(0, len(source_segments), batch_size)
    ]

    async def process_batch(start: int, batch: list[dict]) -> dict[int, dict]:
        expected_indexes = range(start, start + len(batch))
        parsed: dict[int, dict] = {}
        for attempt in range(BATCH_ATTEMPTS):
            # On a retry, only ask again for the segments the model skipped.
            payload_segments = [
                {
                    "index": index,
                    "start": source_segments[index].get("start"),
                    "text": str(source_segments[index].get("text") or ""),
                }
                for index in expected_indexes
                if index not in parsed
            ]
            if not payload_segments:
                break
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
                    thinking=False,
                    model_name=settings.gemini_light_model,
                    list_key="segments",
                    schema=SEGMENTS_SCHEMA,
                )
            parsed.update(_parse_batch(response.get("segments"), expected_indexes, needs_translation))
            if len(parsed) < len(batch):
                logger.warning(
                    "[transcript.process] model skipped segments batch_start=%s missing=%s attempt=%s",
                    start,
                    len(batch) - len(parsed),
                    attempt + 1,
                )
        return parsed

    results = await asyncio.gather(
        *(process_batch(start, batch) for start, batch in batches),
        return_exceptions=True,
    )
    labels: dict[int, dict] = {}
    for batch_result in results:
        if isinstance(batch_result, BaseException):
            raise batch_result
        labels.update(batch_result)

    fallback_count = len(source_segments) - len(labels)
    if fallback_count > max(MIN_FALLBACK_ALLOWANCE, MAX_FALLBACK_SHARE * len(source_segments)):
        raise ValueError(
            f"Gemini transcript processing omitted too many segments ({fallback_count} of {len(source_segments)})"
        )
    if fallback_count:
        logger.warning(
            "[transcript.process] using source text and unknown role for %s of %s segments",
            fallback_count,
            len(source_segments),
        )

    source_output = {**transcript, "segments": []}
    english_output = {
        **transcript,
        "language": "en",
        "source_language": transcript.get("language"),
        "segments": [],
    }
    for index, source_segment in enumerate(source_segments):
        source_text = str(source_segment.get("text") or "")
        label = labels.get(index) or {
            "text": source_text,
            "speaker_role": "unknown",
            "role_confidence": 0.0,
            "role_evidence": "",
        }
        english_text = label["text"] if needs_translation else source_text
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
                "text": english_text,
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
