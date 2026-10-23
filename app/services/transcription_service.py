import av
import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel

from app.core.config import settings

SAMPLE_RATE = 16000
_model = None
_batched_model = None


def model() -> BatchedInferencePipeline:
    global _model, _batched_model

    if _model is None:
        _model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            cpu_threads=settings.whisper_cpu_threads,
        )
        _batched_model = BatchedInferencePipeline(model=_model)

    return _batched_model


def warm_model() -> None:
    model()


def decode_audio_samples(path: str) -> np.ndarray:
    """Decode the media audio once into mono float32 samples at 16 kHz."""
    container = av.open(path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        container.close()
        return np.zeros(0, dtype=np.float32)

    resampler = av.audio.resampler.AudioResampler(
        format="flt",
        layout="mono",
        rate=SAMPLE_RATE,
    )
    chunks = []

    try:
        for frame in container.decode(audio_streams[0]):
            chunks.extend(_resampled_chunks(resampler.resample(frame)))
        chunks.extend(_resampled_chunks(resampler.resample(None)))
    finally:
        container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(chunks).astype(np.float32, copy=False)


def _resampled_chunks(frames) -> list[np.ndarray]:
    return [frame.to_ndarray().reshape(-1).astype(np.float32, copy=False) for frame in frames]


def transcribe_samples(audio: np.ndarray) -> dict:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return {"language": None, "duration": 0.0, "segments": [], "text": ""}

    segments, info = model().transcribe(
        audio,
        language=settings.whisper_language or None,
        beam_size=settings.whisper_beam_size,
        batch_size=settings.whisper_batch_size,
        vad_filter=True,
        word_timestamps=False,
    )
    rows = [
        {
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
            "speaker_role": "unknown",
            "confidence": getattr(segment, "avg_logprob", None),
        }
        for segment in segments
    ]
    return {
        "language": info.language,
        "duration": info.duration,
        "segments": rows,
        "text": " ".join(row["text"] for row in rows),
    }


def transcribe(path: str) -> dict:
    return transcribe_samples(decode_audio_samples(path))
