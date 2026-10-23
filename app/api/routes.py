import logging

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool
from app.api.dependencies import require_internal_token
from app.models.schemas import AnalyzeRequest, StageRequest
from app.services.r2_storage import downloaded_object_file
from app.services.transcription_service import transcribe
from app.services.transcription_service import decode_audio_samples
from app.services.transcript_processing_service import translate_and_label
from app.services.teaching_service import score
from app.services.audio_quality_service import analyze as analyze_audio
from app.services.video_screen_service import analyze as analyze_video
from app.services.pipeline_manager import run

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def root():
    return {"success":True,"data":{"service":"Humaniti Video Analysis","status":"running"}}

@router.get("/health")
async def health():
    return {"success":True,"data":{"status":"ok"}}

@router.post("/api/v1/analyze")
async def analyze(request: AnalyzeRequest, _=Depends(require_internal_token)):
    try:
        data = await run(request.object_key,request.evaluation_id)
        return {"success":True,"message":"Analysis completed","data":data}
    except LookupError as exc:
        raise HTTPException(status_code=404,detail="Evaluation not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409,detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "[api.analyze] video analysis failed evaluation_id=%s",
            request.evaluation_id
        )
        raise HTTPException(status_code=500,detail="Video analysis failed") from exc

@router.post("/api/v1/test/transcription")
async def test_transcription(request: StageRequest, _=Depends(require_internal_token)):
    try:
        with downloaded_object_file(request.object_key) as path:
            transcript = await run_in_threadpool(transcribe,path)
            transcript, _, _, speakers = await translate_and_label(transcript)
            return {"success":True,"data":{"transcription":transcript,"speaker_identification":speakers}}
    except Exception as exc:
        logger.exception("[api.test.transcription] transcription test failed")
        raise HTTPException(status_code=500,detail="Transcription test failed") from exc

@router.post("/api/v1/test/teaching-effectiveness")
async def test_teaching(request: StageRequest, _=Depends(require_internal_token)):
    try:
        with downloaded_object_file(request.object_key) as path:
            transcript = await run_in_threadpool(transcribe,path)
            transcript, english_transcript, _, _ = await translate_and_label(transcript)
            result = await run_in_threadpool(score,english_transcript)
            return {"success":True,"data":{"transcription":transcript,"teaching_effectiveness":result}}
    except Exception as exc:
        logger.exception("[api.test.teaching-effectiveness] teaching test failed")
        raise HTTPException(status_code=500,detail="Teaching effectiveness test failed") from exc

@router.post("/api/v1/test/audio-quality")
async def test_audio(request: StageRequest, _=Depends(require_internal_token)):
    try:
        with downloaded_object_file(request.object_key) as path:
            samples = await run_in_threadpool(decode_audio_samples, path)
            result = await run_in_threadpool(analyze_audio, samples, 16000)
            return {"success":True,"data":result}
    except Exception as exc:
        logger.exception("[api.test.audio-quality] audio quality test failed")
        raise HTTPException(status_code=500,detail="Audio quality test failed") from exc

@router.post("/api/v1/test/video-screen")
async def test_video(request: StageRequest, _=Depends(require_internal_token)):
    try:
        with downloaded_object_file(request.object_key) as path:
            transcript = await run_in_threadpool(transcribe,path)
            _, english_transcript, _, _ = await translate_and_label(transcript)
            result = await run_in_threadpool(analyze_video,path,english_transcript)
            return {"success":True,"data":result}
    except Exception as exc:
        logger.exception("[api.test.video-screen] video screen test failed")
        raise HTTPException(status_code=500,detail="Video screen test failed") from exc
