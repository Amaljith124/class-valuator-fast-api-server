import asyncio
import threading
import time

import pytest

from app.services import transcript_processing_service as processing


def _transcript(language="ml", count=2):
    segments = [
        {"start": float(index), "end": float(index + 1), "text": f"source {index}", "confidence": -0.1}
        for index in range(count)
    ]
    return {
        "language": language,
        "duration": float(count),
        "text": " ".join(segment["text"] for segment in segments),
        "segments": segments,
    }


def test_translate_and_role_label_merges_results_and_preserves_source(monkeypatch):
    selected_models = []

    def fake_generate_json(instructions, payload, thinking=None, model_name=None, **kwargs):
        selected_models.append(model_name)
        return {
            "segments": [
                {
                    "index": segment["index"],
                    "text": f"English {segment['index']}",
                    "speaker_role": "teacher",
                    "confidence": 0.9,
                }
                for segment in payload["segments"]
            ]
        }

    monkeypatch.setattr(processing, "generate_json", fake_generate_json)
    source, english, translated, roles = asyncio.run(processing.translate_and_label(_transcript()))

    assert translated is True
    assert source["segments"][0]["text"] == "source 0"
    assert source["segments"][0]["speaker_role"] == "teacher"
    assert english["segments"][0]["text"] == "English 0"
    assert english["segments"][0]["role_evidence"] == ""
    assert roles["segments"][1]["speaker_role"] == "teacher"
    assert selected_models == ["gemini-3.5-flash-lite"]


def test_english_transcript_keeps_original_text_and_labels_roles(monkeypatch):
    def fake_generate_json(instructions, payload, thinking=None, model_name=None, **kwargs):
        return {
            "segments": [
                {
                    "index": segment["index"],
                    "text": "model altered text",
                    "speaker_role": "unknown",
                    "confidence": 0.4,
                    "reason": "No clear role cue",
                }
                for segment in payload["segments"]
            ]
        }

    monkeypatch.setattr(processing, "generate_json", fake_generate_json)
    source, english, translated, _ = asyncio.run(
        processing.translate_and_label(_transcript(language="en"))
    )

    assert translated is False
    assert english["segments"][0]["text"] == "source 0"
    assert source["segments"][0]["role_evidence"] == "No clear role cue"


def test_one_skipped_segment_is_retried_then_falls_back_without_failing(monkeypatch):
    monkeypatch.setattr(processing.settings, "llm_batch_size", 10)
    monkeypatch.setattr(processing, "_llm_semaphore", asyncio.Semaphore(3))
    requested = []

    def fake_generate_json(instructions, payload, thinking=None, model_name=None, **kwargs):
        requested.append([segment["index"] for segment in payload["segments"]])
        # The model never returns index 1, even when asked again.
        return {
            "segments": [
                {"index": segment["index"], "text": f"English {segment['index']}", "speaker_role": "teacher", "confidence": 0.9}
                for segment in payload["segments"]
                if segment["index"] != 1
            ]
        }

    monkeypatch.setattr(processing, "generate_json", fake_generate_json)
    source, english, _, _ = asyncio.run(processing.translate_and_label(_transcript(count=4)))

    assert requested == [[0, 1, 2, 3], [1]]
    assert english["segments"][1]["text"] == "source 1"
    assert english["segments"][1]["speaker_role"] == "unknown"
    assert english["segments"][0]["text"] == "English 0"
    assert source["segments"][1]["speaker_role"] == "unknown"


def test_duplicate_indexes_keep_the_first_answer(monkeypatch):
    monkeypatch.setattr(processing, "_llm_semaphore", asyncio.Semaphore(3))
    monkeypatch.setattr(
        processing,
        "generate_json",
        lambda *args, **kwargs: {
            "segments": [
                {"index": 0, "text": "First", "speaker_role": "teacher", "confidence": 0.9},
                {"index": 0, "text": "Duplicate", "speaker_role": "student", "confidence": 0.9},
                {"index": 1, "text": "Second", "speaker_role": "student", "confidence": 0.9},
            ]
        },
    )

    _, english, _, _ = asyncio.run(processing.translate_and_label(_transcript(count=2)))

    assert english["segments"][0]["text"] == "First"
    assert english["segments"][0]["speaker_role"] == "teacher"


def test_mostly_missing_output_is_rejected(monkeypatch):
    monkeypatch.setattr(processing.settings, "llm_batch_size", 50)
    monkeypatch.setattr(processing, "_llm_semaphore", asyncio.Semaphore(3))
    monkeypatch.setattr(processing, "generate_json", lambda *args, **kwargs: {"segments": []})

    with pytest.raises(ValueError, match="omitted too many segments"):
        asyncio.run(processing.translate_and_label(_transcript(count=20)))


def test_unusual_roles_confidence_and_indexes_are_normalised_not_fatal(monkeypatch):
    monkeypatch.setattr(processing, "_llm_semaphore", asyncio.Semaphore(3))
    monkeypatch.setattr(
        processing,
        "generate_json",
        lambda *args, **kwargs: {
            "segments": [
                {"index": "0", "text": "A", "speaker_role": " Teacher ", "confidence": "high"},
                {"index": 1.0, "text": "B", "speaker_role": "moderator", "confidence": 7},
                "not a dict",
                {"index": 99, "text": "ignored", "speaker_role": "teacher", "confidence": 1},
            ]
        },
    )

    _, english, _, _ = asyncio.run(processing.translate_and_label(_transcript(count=2)))

    assert english["segments"][0]["speaker_role"] == "teacher"
    assert english["segments"][0]["role_confidence"] == 0.0
    assert english["segments"][1]["speaker_role"] == "unknown"
    assert english["segments"][1]["role_confidence"] == 1.0


def test_english_transcript_does_not_require_model_text(monkeypatch):
    monkeypatch.setattr(processing, "_llm_semaphore", asyncio.Semaphore(3))
    monkeypatch.setattr(
        processing,
        "generate_json",
        lambda *args, **kwargs: {
            "segments": [
                {"index": 0, "speaker_role": "teacher", "confidence": 0.9},
                {"index": 1, "speaker_role": "student", "confidence": 0.9},
            ]
        },
    )

    _, english, translated, _ = asyncio.run(processing.translate_and_label(_transcript(language="en")))

    assert translated is False
    assert [segment["text"] for segment in english["segments"]] == ["source 0", "source 1"]


def test_transcript_without_segments_needs_no_model_call(monkeypatch):
    monkeypatch.setattr(processing, "generate_json", lambda *a, **k: pytest.fail("called"))

    source, english, _, roles = asyncio.run(
        processing.translate_and_label({"language": None, "duration": 0.0, "segments": [], "text": ""})
    )

    assert source["segments"] == [] and english["segments"] == [] and english["text"] == ""
    assert roles["segments"] == []


def test_concurrent_batches_are_reassembled_in_transcript_order(monkeypatch):
    monkeypatch.setattr(processing.settings, "llm_batch_size", 1)
    monkeypatch.setattr(processing, "_llm_semaphore", asyncio.Semaphore(3))
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_generate_json(instructions, payload, thinking=None, model_name=None, **kwargs):
        nonlocal active, maximum_active
        index = payload["segments"][0]["index"]
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.10 if index == 0 else 0.01)
        with lock:
            active -= 1
        return {
            "segments": [
                {
                    "index": index,
                    "text": f"English {index}",
                    "speaker_role": "student",
                    "confidence": 0.8,
                }
            ]
        }

    monkeypatch.setattr(processing, "generate_json", fake_generate_json)
    _, english, _, _ = asyncio.run(processing.translate_and_label(_transcript(count=4)))

    assert [segment["text"] for segment in english["segments"]] == [
        "English 0", "English 1", "English 2", "English 3"
    ]
    assert maximum_active > 1
    assert maximum_active <= 3
