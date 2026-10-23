import json
from app.database.db import execute, fetch_one

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
        (json.dumps(checkpoint), evaluation_id),
    )

async def save_result(evaluation_id: str, transcript: str, transcript_english: str, result: dict) -> None:
    await execute("""UPDATE evaluation SET status='completed',transcript=%s,transcript_english=%s,evaluation_result=%s::jsonb,
        ai_provider='Google Gemini',ai_model=%s,error_message=NULL,updated_date=NOW(),completed_date=NOW()
        WHERE evaluation_id=%s""", (transcript, transcript_english, json.dumps(result), result.get("model"), evaluation_id))

async def mark_failed(evaluation_id: str, message: str) -> None:
    await execute("UPDATE evaluation SET status='failed',error_message=%s,updated_date=NOW(),completed_date=NOW() WHERE evaluation_id=%s", (message, evaluation_id))
