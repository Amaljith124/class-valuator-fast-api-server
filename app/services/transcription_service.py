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


def _speech_windows(audio: np.ndarray) -> list[np.ndarray]:
    """Up to N windows of real speech spread across the recording, found with the VAD."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    window = settings.whisper_detection_window_seconds * SAMPLE_RATE
    spans = get_speech_timestamps(audio, VadOptions(threshold=settings.whisper_vad_threshold))
    speech = np.concatenate([audio[span["start"]:span["end"]] for span in spans]) if spans else audio
    if speech.size <= window:
        return [speech]
    count = max(1, settings.whisper_detection_windows)
    starts = np.linspace(0, speech.size - window, count).astype(int)
    return [speech[start:start + window] for start in starts]


def detect_language(audio: np.ndarray) -> str | None:
    """Pick the class language from several speech windows, limited to the candidate languages.

    Whisper's default looks only at the first 30 seconds, which for a class is often waiting
    noise, and then transcribes the whole recording in the wrongly guessed language.
    """
    candidates = [code.strip().lower() for code in settings.whisper_candidate_languages.split(",") if code.strip()]
    totals: dict[str, float] = {}
    model()
    for window in _speech_windows(audio):
        _, _, probabilities = _model.detect_language(audio=window)
        for code, probability in probabilities:
            if not candidates or code in candidates:
                totals[code] = totals.get(code, 0.0) + float(probability)
    if not totals:
        return None
    return max(totals, key=totals.get)


def transcribe_samples(audio: np.ndarray) -> dict:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return {"language": None, "duration": 0.0, "segments": [], "text": ""}

    language = settings.whisper_language or detect_language(audio)
    segments, info = model().transcribe(
        audio,
        language=language,
        beam_size=settings.whisper_beam_size,
        batch_size=settings.whisper_batch_size,
        vad_filter=True,
        vad_parameters={
            "threshold": settings.whisper_vad_threshold,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": settings.whisper_min_silence_ms,
        },
        temperature=[0.0, 0.2, 0.4],
        compression_ratio_threshold=settings.whisper_compression_ratio_threshold,
        log_prob_threshold=settings.whisper_log_prob_threshold,
        no_speech_threshold=settings.whisper_no_speech_threshold,
        word_timestamps=False,
    )
    rows = []
    for segment in segments:
        compression_ratio = getattr(segment, "compression_ratio", None)
        no_speech_prob = getattr(segment, "no_speech_prob", None)
        avg_logprob = getattr(segment, "avg_logprob", None)
        looks_like_noise = (
            isinstance(compression_ratio, (int, float))
            and compression_ratio > settings.whisper_compression_ratio_threshold
        ) or (
            isinstance(no_speech_prob, (int, float)) and isinstance(avg_logprob, (int, float))
            and no_speech_prob > settings.whisper_no_speech_threshold
            and avg_logprob < settings.whisper_log_prob_threshold
        )
        if looks_like_noise:
            continue
        rows.append(
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "speaker_role": "unknown",
                "confidence": avg_logprob,
            }
        )
    return {
        "language": info.language,
        "duration": info.duration,
        "segments": rows,
        "text": " ".join(row["text"] for row in rows),
    }


def transcribe(path: str) -> dict:
    return transcribe_samples(decode_audio_samples(path))
