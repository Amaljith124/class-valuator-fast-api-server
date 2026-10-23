import json
import math

from app.database.db import execute, fetch_one


def _json_safe(value):
    """Postgres jsonb rejects NaN/Infinity and NUL characters; clean them before saving."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, str):
        return value.replace("\x00", "")
    return value


def _json_default(value):
    return value.item() if hasattr(value, "item") else str(value)


def _dumps(value) -> str:
    return json.dumps(_json_safe(value), default=_json_default)


def _clean_text(value) -> str:
    return str(value or "").replace("\x00", "")


async def get_evaluation(evaluation_id: str):
    return await fetch_one(
        "SELECT evaluation_id, status, evaluation_result FROM evaluation WHERE evaluation_id=%s",
        (evaluation_id,),
    )

async def mark_processing(evaluation_id: str) -> None:
    await execute("UPDATE evaluation SET status='processing',updated_date=NOW() WHERE evaluation_id=%s", (evaluation_id,))


async def save_transcript_checkpoint(evaluation_id: str, transcript: dict) -> None:
    checkpoint = {"_pipeline_checkpoint": {"transcript": transcript}}
    await execute(
        "UPDATE evaluation SET evaluation_result=%s::jsonb,updated_date=NOW() WHERE evaluation_id=%s",
        (_dumps(checkpoint), evaluation_id),
    )

async def save_result(evaluation_id: str, transcript: str, transcript_english: str, result: dict) -> None:
    await execute("""UPDATE evaluation SET status='completed',transcript=%s,transcript_english=%s,evaluation_result=%s::jsonb,
        ai_provider='Google Gemini',ai_model=%s,error_message=NULL,updated_date=NOW(),completed_date=NOW()
        WHERE evaluation_id=%s""", (_clean_text(transcript), _clean_text(transcript_english), _dumps(result), result.get("model"), evaluation_id))

async def mark_failed(evaluation_id: str, message: str) -> None:
    await execute("UPDATE evaluation SET status='failed',error_message=%s,updated_date=NOW(),completed_date=NOW() WHERE evaluation_id=%s", (message, evaluation_id))
