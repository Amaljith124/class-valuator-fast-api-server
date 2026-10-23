import numpy as np

from app.services.audio_quality_service import analyze


def test_audio_quality_reports_mono_16khz_metrics_for_synthetic_signal():
    sample_rate = 16000
    duration_seconds = 2
    time_values = np.arange(sample_rate * duration_seconds) / sample_rate
    samples = (0.5 * np.sin(2 * np.pi * 440 * time_values)).astype(np.float32)

    result = analyze(samples, sample_rate)

    assert result["assessable"] is True
    assert 0 <= result["score"] <= 100
    assert result["metrics"]["sample_rate_hz"] == 16000
    assert result["metrics"]["channels"] == 1
    assert result["metrics"]["duration_seconds"] == 2.0
    assert result["metrics"]["clipping_ratio"] == 0.0
    assert result["metrics"]["rms_dbfs"] == -9.03
