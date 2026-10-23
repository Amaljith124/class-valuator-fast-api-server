import asyncio
import contextlib

import numpy as np
import pytest

from app.services import pipeline_manager as pipeline
from app.services.teaching_service import CRITERIA

EVALUATION_ID = "11111111-1111-1111-1111-111111111111"


def _transcript(texts=("Welcome to fractions", "Any questions")):
    segments = [
        {"start": float(index), "end": float(index + 1), "text": text, "speaker_role": "unknown"}
        for index, text in enumerate(texts)
    ]
    return {"language": "en", "duration": float(len(segments)), "segments": segments, "text": " ".join(texts)}


def _criteria(score=80, quote="Welcome to fractions"):
    return {name: {"score": score, "assessable": True, "evidence": [quote]} for name in CRITERIA}


class _Recorder:
    def __init__(self):
        self.saved = []
        self.failed = []
        self.transcribed = 0


@pytest.fixture
def stages(monkeypatch):
    recorder = _Recorder()

    @contextlib.contextmanager
    def fake_download(object_key):
        yield "/tmp/fake-recording.mp4"

    async def fake_get_evaluation(evaluation_id):
        return {"evaluation_id": evaluation_id, "status": "pending", "evaluation_result": None}

    async def fake_noop(*args, **kwargs):
        return None

    async def fake_save_result(evaluation_id, transcript, transcript_english, result):
        recorder.saved.append((evaluation_id, transcript, transcript_english, result))

    async def fake_mark_failed(evaluation_id, message):
        recorder.failed.append((evaluation_id, message))

    def fake_transcribe(samples):
        recorder.transcribed += 1
        return _transcript()

    async def fake_translate_and_label(transcript):
        english = {**transcript, "segments": [dict(segment) for segment in transcript["segments"]]}
        return transcript, english, False, {"limitations": []}

    monkeypatch.setattr(pipeline, "downloaded_object_file", fake_download)
    monkeypatch.setattr(pipeline, "get_evaluation", fake_get_evaluation)
    monkeypatch.setattr(pipeline, "mark_processing", fake_noop)
    monkeypatch.setattr(pipeline, "save_transcript_checkpoint", fake_noop)
    monkeypatch.setattr(pipeline, "save_result", fake_save_result)
    monkeypatch.setattr(pipeline, "mark_failed", fake_mark_failed)
    monkeypatch.setattr(pipeline, "decode_audio_samples", lambda path: np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(pipeline, "transcribe_samples", fake_transcribe)
    monkeypatch.setattr(pipeline, "analyze_audio", lambda samples: {"assessable": True, "score": 90})
    monkeypatch.setattr(pipeline, "translate_and_label", fake_translate_and_label)
    monkeypatch.setattr(pipeline, "score", lambda transcript: {"criteria": _criteria(), "lesson_summary": "s"})
    monkeypatch.setattr(pipeline, "analyze_video", lambda path, transcript: {"assessable": True, "score": 70, "observations": []})
    monkeypatch.setattr(pipeline, "_whisper_semaphore", asyncio.Semaphore(1))
    return recorder


def _run():
    return asyncio.run(pipeline.run("recordings/2026-10-23/a.mp4", EVALUATION_ID))


def test_successful_run_saves_a_completed_result(stages):
    outcome = _run()

    assert outcome["status"] == "completed"
    assert outcome["result"]["overall_score"] == 80.0
    assert len(stages.saved) == 1 and stages.failed == []


def test_video_failure_does_not_fail_the_job(stages, monkeypatch):
    def broken_video(path, transcript):
        raise ValueError("Gemini response must be a JSON object")

    monkeypatch.setattr(pipeline, "analyze_video", broken_video)

    outcome = _run()

    video = outcome["result"]["video_screen_analysis"]
    assert outcome["status"] == "completed"
    assert video["assessable"] is False and video["score"] is None
    assert outcome["result"]["overall_score"] == 80.0
    assert stages.failed == []


def test_teaching_failure_fails_the_job_and_marks_it_failed(stages, monkeypatch):
    def broken_score(transcript):
        raise ValueError("boom")

    monkeypatch.setattr(pipeline, "score", broken_score)

    with pytest.raises(ValueError, match="boom"):
        _run()

    assert stages.failed == [(EVALUATION_ID, "Analysis failed; check service logs.")]
    assert stages.saved == []


def test_failure_to_record_the_failure_does_not_hide_the_real_error(stages, monkeypatch):
    async def broken_mark_failed(evaluation_id, message):
        raise RuntimeError("database is down")

    monkeypatch.setattr(pipeline, "score", lambda transcript: (_ for _ in ()).throw(ValueError("teaching broke")))
    monkeypatch.setattr(pipeline, "mark_failed", broken_mark_failed)

    with pytest.raises(ValueError, match="teaching broke"):
        _run()


def test_already_completed_evaluation_is_rejected_without_marking_it_failed(stages, monkeypatch):
    async def completed(evaluation_id):
        return {"evaluation_id": evaluation_id, "status": "completed", "evaluation_result": None}

    monkeypatch.setattr(pipeline, "get_evaluation", completed)

    with pytest.raises(pipeline.EvaluationAlreadyCompleted):
        _run()

    assert stages.failed == [] and stages.transcribed == 0


def test_unknown_evaluation_raises_lookup_error(stages, monkeypatch):
    async def missing(evaluation_id):
        return None

    monkeypatch.setattr(pipeline, "get_evaluation", missing)

    with pytest.raises(LookupError):
        _run()


def test_saved_transcript_checkpoint_skips_transcription(stages, monkeypatch):
    async def with_checkpoint(evaluation_id):
        return {
            "evaluation_id": evaluation_id,
            "status": "failed",
            "evaluation_result": {"_pipeline_checkpoint": {"transcript": _transcript()}},
        }

    monkeypatch.setattr(pipeline, "get_evaluation", with_checkpoint)

    outcome = _run()

    assert outcome["status"] == "completed"
    assert stages.transcribed == 0


def test_transcription_failure_is_reported_as_a_failed_job(stages, monkeypatch):
    def broken(samples):
        raise RuntimeError("whisper crashed")

    monkeypatch.setattr(pipeline, "transcribe_samples", broken)

    with pytest.raises(RuntimeError):
        _run()

    assert len(stages.failed) == 1


def test_recording_without_speech_completes_with_everything_unassessable(stages, monkeypatch):
    from app.services import teaching_service, transcript_processing_service as processing

    silent = {"language": None, "duration": 0.0, "segments": [], "text": ""}
    monkeypatch.setattr(pipeline, "transcribe_samples", lambda samples: silent)
    monkeypatch.setattr(pipeline, "translate_and_label", processing.translate_and_label)
    monkeypatch.setattr(pipeline, "score", teaching_service.score)

    outcome = _run()

    assert outcome["status"] == "completed"
    assert outcome["result"]["overall_score"] is None
    assert all(not item["assessable"] for item in outcome["result"]["criteria"].values())
