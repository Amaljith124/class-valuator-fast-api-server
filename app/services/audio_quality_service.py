import numpy as np

from app.services.transcription_service import SAMPLE_RATE


def analyze(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return {"assessable": False, "score": None, "reason": "No audio samples", "metrics": {}}

    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono**2)))
    clipping = float(np.mean(np.abs(mono) >= 0.999))
    silence = float(np.mean(np.abs(mono) < 0.003))
    rms_dbfs = float(20 * np.log10(max(rms, 1e-8)))
    score = max(
        0,
        min(100, round(100 - min(clipping * 1000, 40) - min(silence * 50, 35) - max(0, abs(rms_dbfs + 20) - 20), 1)),
    )
    # Audio decoding normalizes to mono float32 at 16 kHz before either analysis runs.
    return {
        "assessable": True,
        "score": score,
        "metrics": {
            "sample_rate_hz": int(sample_rate),
            "channels": 1,
            "duration_seconds": round(len(mono) / sample_rate, 3),
            "peak_amplitude": round(peak, 5),
            "rms_dbfs": round(rms_dbfs, 2),
            "clipping_ratio": round(clipping, 6),
            "silence_ratio": round(silence, 5),
        },
        "method": "Signal-derived heuristics; not a human intelligibility test",
        "warnings": [
            "Silence may represent pauses. These measurements do not establish intelligibility."
        ],
    }
