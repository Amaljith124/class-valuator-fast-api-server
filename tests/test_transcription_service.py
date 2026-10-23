from types import SimpleNamespace

import numpy as np
import pytest

from app.services import transcription_service as transcription


class _FakeModel:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = 0

    def detect_language(self, audio=None, **kwargs):
        self.calls += 1
        return "xx", 0.0, self.probabilities.pop(0) if len(self.probabilities) > 1 else self.probabilities[0]


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(transcription, "model", lambda: None)
    monkeypatch.setattr(transcription, "_speech_windows", lambda audio: [audio, audio, audio])


def test_language_is_chosen_from_several_windows_within_the_candidates(monkeypatch, windows):
    fake = _FakeModel([[("ml", 0.5), ("en", 0.4), ("fr", 0.1)], [("en", 0.8), ("ml", 0.1)], [("en", 0.7), ("fr", 0.2)]])
    monkeypatch.setattr(transcription, "_model", fake, raising=False)

    assert transcription.detect_language(np.zeros(10, dtype=np.float32)) == "en"
    assert fake.calls == 3


def test_languages_outside_the_candidate_list_are_ignored(monkeypatch, windows):
    fake = _FakeModel([[("fr", 0.9), ("ml", 0.05)]])
    monkeypatch.setattr(transcription, "_model", fake, raising=False)

    assert transcription.detect_language(np.zeros(10, dtype=np.float32)) == "ml"


def test_no_candidate_probabilities_gives_none(monkeypatch, windows):
    monkeypatch.setattr(transcription, "_model", _FakeModel([[("fr", 0.9)]]), raising=False)

    assert transcription.detect_language(np.zeros(10, dtype=np.float32)) is None


def _segment(text, start=0.0, compression=1.2, no_speech=0.1, logprob=-0.3):
    return SimpleNamespace(
        start=start, end=start + 2, text=text,
        compression_ratio=compression, no_speech_prob=no_speech, avg_logprob=logprob,
    )


def _pipeline(monkeypatch, segments, seen):
    class Pipeline:
        def transcribe(self, audio, **kwargs):
            seen.update(kwargs)
            return iter(segments), SimpleNamespace(language=kwargs["language"], duration=60.0)

    monkeypatch.setattr(transcription, "model", lambda: Pipeline())


def test_noisy_segments_are_dropped_and_good_ones_kept(monkeypatch):
    seen = {}
    segments = [
        _segment(" good speech "),
        _segment("loop loop loop", start=3, compression=3.5),
        _segment("hallucination in silence", start=6, no_speech=0.9, logprob=-1.8),
        _segment("quiet but real", start=9, no_speech=0.9, logprob=-0.2),
    ]
    _pipeline(monkeypatch, segments, seen)
    monkeypatch.setattr(transcription.settings, "whisper_language", "en")

    result = transcription.transcribe_samples(np.ones(100, dtype=np.float32))

    assert [row["text"] for row in result["segments"]] == ["good speech", "quiet but real"]
    assert result["text"] == "good speech quiet but real"


def test_a_configured_language_skips_detection_and_decoding_is_constrained(monkeypatch):
    seen = {}
    _pipeline(monkeypatch, [], seen)
    monkeypatch.setattr(transcription.settings, "whisper_language", "ml")
    monkeypatch.setattr(transcription, "detect_language", lambda audio: pytest.fail("must not detect"))

    transcription.transcribe_samples(np.ones(100, dtype=np.float32))

    assert seen["language"] == "ml"
    assert seen["compression_ratio_threshold"] == transcription.settings.whisper_compression_ratio_threshold
    assert seen["temperature"] == [0.0, 0.2, 0.4]
    assert seen["vad_parameters"]["min_silence_duration_ms"] == transcription.settings.whisper_min_silence_ms


def test_detected_language_is_used_when_none_is_configured(monkeypatch):
    seen = {}
    _pipeline(monkeypatch, [], seen)
    monkeypatch.setattr(transcription.settings, "whisper_language", None)
    monkeypatch.setattr(transcription, "detect_language", lambda audio: "en")

    transcription.transcribe_samples(np.ones(100, dtype=np.float32))

    assert seen["language"] == "en"
