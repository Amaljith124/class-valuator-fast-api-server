import asyncio
import json
import logging
import time

from app.core.config import settings
from app.core.timing import timed_stage
from app.database.repository import (
    get_evaluation,
    mark_failed,
    mark_processing,
    save_result,
    save_transcript_checkpoint,
)
from app.services.audio_quality_service import analyze as analyze_audio
from app.services.evidence_service import combine
from app.services.r2_storage import downloaded_object_file
from app.services.teaching_service import score
from app.services.transcription_service import decode_audio_samples, transcribe_samples
from app.services.transcript_processing_service import translate_and_label
from app.services.video_screen_service import analyze as analyze_video

logger = logging.getLogger(__name__)
_whisper_semaphore = asyncio.Semaphore(settings.whisper_concurrency)


async def _timed_blocking(stage: str, evaluation_id: str | None, function, *args):
    async with timed_stage(logger, stage, evaluation_id):
        return await asyncio.to_thread(function, *args)


async def _timed_whisper(samples, evaluation_id: str | None):
    async with timed_stage(logger, "transcription", evaluation_id):
        async with _whisper_semaphore:
            return await asyncio.to_thread(transcribe_samples, samples)


async def _transcribe_and_checkpoint(samples, evaluation_id: str | None):
    try:
        transcript = await _timed_whisper(samples, evaluation_id)
    except Exception as error:
        raise RuntimeError("Pipeline stage failed: transcription") from error
    if evaluation_id and settings.transcript_checkpoint_enabled:
        try:
            async with timed_stage(logger, "save transcript checkpoint", evaluation_id):
                await save_transcript_checkpoint(evaluation_id, transcript)
        except Exception as error:
            raise RuntimeError("Pipeline stage failed: save transcript checkpoint") from error
    return transcript


async def _parallel_stages(stages: list[tuple[str, object]], evaluation_id: str | None):
    results = await asyncio.gather(
        *(
            _timed_blocking(name, evaluation_id, function, *args)
            for name, function, args in stages
        ),
        return_exceptions=True,
    )
    for (name, _, _), result in zip(stages, results):
        if isinstance(result, BaseException):
            raise RuntimeError(f"Pipeline stage failed: {name}") from result
    return results


def _checkpoint_from_result(evaluation_result):
    if isinstance(evaluation_result, str):
        try:
            evaluation_result = json.loads(evaluation_result)
        except json.JSONDecodeError:
            return None
    if not isinstance(evaluation_result, dict):
        return None
    checkpoint = evaluation_result.get("_pipeline_checkpoint", {})
    transcript = checkpoint.get("transcript") if isinstance(checkpoint, dict) else None
    return transcript if isinstance(transcript, dict) else None


async def run(object_key: str, evaluation_id: str | None) -> dict:
    total_started_at = time.perf_counter()
    stage = "load evaluation"
    download_manager = None
    download_context_entered = False
    evaluation_started = False

    try:
        checkpoint = None
        if evaluation_id:
            async with timed_stage(logger, stage, evaluation_id):
                row = await get_evaluation(evaluation_id)
            if row is None:
                raise LookupError("Evaluation not found")
            if row["status"] == "completed":
                raise ValueError("Evaluation is already completed")
            checkpoint = _checkpoint_from_result(row.get("evaluation_result"))
            async with timed_stage(logger, "mark evaluation processing", evaluation_id):
                await mark_processing(evaluation_id)
                evaluation_started = True

        stage = "download recording from R2"
        async with timed_stage(logger, stage, evaluation_id):
            download_manager = downloaded_object_file(object_key)
            path = await asyncio.to_thread(download_manager.__enter__)
            download_context_entered = True

        try:
            stage = "decode audio"
            samples = await _timed_blocking(
                stage,
                evaluation_id,
                decode_audio_samples,
                path,
            )

            if checkpoint is not None:
                transcript = checkpoint
                stages = [
                    ("audio quality analysis", analyze_audio, (samples,)),
                ]
                stage = "audio quality analysis"
                outcomes = await _parallel_stages(stages, evaluation_id)
                audio = outcomes[0]
            else:
                stage = "transcription and audio quality analysis"
                outcomes = await asyncio.gather(
                    _transcribe_and_checkpoint(samples, evaluation_id),
                    _timed_blocking(
                        "audio quality analysis",
                        evaluation_id,
                        analyze_audio,
                        samples,
                    ),
                    return_exceptions=True,
                )
                transcript_result, audio_result = outcomes
                if isinstance(transcript_result, BaseException):
                    stage = (
                        str(transcript_result).removeprefix("Pipeline stage failed: ")
                        if str(transcript_result).startswith("Pipeline stage failed: ")
                        else "transcription"
                    )
                    raise transcript_result
                if isinstance(audio_result, BaseException):
                    stage = "audio quality analysis"
                    raise audio_result
                transcript = transcript_result
                audio = audio_result

            stage = "English translation and speaker role inference"
            async with timed_stage(logger, stage, evaluation_id):
                transcript, english_transcript, was_translated, roles = await translate_and_label(transcript)

            stage = "teaching score and video screen analysis"
            teaching_result, video_result = await asyncio.gather(
                _timed_blocking(
                    "teaching effectiveness scoring",
                    evaluation_id,
                    score,
                    english_transcript,
                ),
                _timed_blocking(
                    "video screen analysis",
                    evaluation_id,
                    analyze_video,
                    path,
                    english_transcript,
                ),
                return_exceptions=True,
            )
            if isinstance(teaching_result, BaseException):
                stage = "teaching effectiveness scoring"
                raise teaching_result
            if isinstance(video_result, BaseException):
                stage = "video screen analysis"
                raise video_result
            teaching = teaching_result
            video = video_result

            stage = "combine analysis results"
            result = await _timed_blocking(
                stage,
                evaluation_id,
                combine,
                teaching,
                audio,
                video,
                english_transcript,
            )
            result.update({"provider": "Google Gemini", "model": settings.gemini_model})
            result["speaker_notes"] = roles.get("limitations", [])
            result["transcription"] = {
                "language": transcript.get("language"),
                "duration": transcript.get("duration"),
                "segments": transcript.get("segments", []),
            }
            result["transcription_english"] = {
                "language": "en",
                "source_language": transcript.get("language"),
                "translated": was_translated,
                "text": english_transcript.get("text", ""),
                "segments": english_transcript.get("segments", []),
            }

            if evaluation_id:
                stage = "save evaluation results"
                async with timed_stage(logger, stage, evaluation_id):
                    await save_result(
                        evaluation_id,
                        transcript.get("text", ""),
                        english_transcript.get("text", ""),
                        result,
                    )

            return {
                "evaluation_id": evaluation_id,
                "status": "completed",
                "provider": "Google Gemini",
                "model": settings.gemini_model,
                "transcript": transcript.get("text", ""),
                "transcript_english": english_transcript.get("text", ""),
                "result": result,
            }
        finally:
            if download_context_entered:
                async with timed_stage(logger, "clean up downloaded recording", evaluation_id):
                    await asyncio.to_thread(download_manager.__exit__, None, None, None)
    except Exception:
        logger.exception(
            "[pipeline.run] analysis failed during stage=%s evaluation_id=%s",
            stage,
            evaluation_id,
        )
        if evaluation_id and evaluation_started:
            await mark_failed(evaluation_id, "Analysis failed; check service logs.")
        raise
    finally:
        logger.info(
            "[pipeline.timing] stage=total job evaluation_id=%s seconds=%.2f",
            evaluation_id or "none",
            time.perf_counter() - total_started_at,
        )
