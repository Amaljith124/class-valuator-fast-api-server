import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.services.gemini_service import analyze_images
from app.templates.prompt_templates import VIDEO_SCREEN_ANALYSIS_INSTRUCTIONS


def _video_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", path,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return 0.0
    return float(json.loads(result.stdout).get("format", {}).get("duration", 0.0))


def _extract_frame(path: str, timestamp: float, folder: str, index: int):
    frame_path = os.path.join(folder, f"frame-{index:04d}.jpg")
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-ss", str(timestamp), "-i", path,
            "-frames:v", "1", "-vf", "scale=1024:-1", "-q:v", "4", frame_path,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0 or not os.path.exists(frame_path):
        return None
    with open(frame_path, "rb") as stream:
        return timestamp, stream.read()


def frames(path: str, interval: int | None = None, maximum: int | None = None) -> list[tuple[float, bytes]]:
    interval = interval or settings.frame_interval_seconds
    maximum = min(maximum or settings.max_video_frames, settings.max_video_frames)
    duration = _video_duration(path)
    timestamps = [
        float(timestamp)
        for timestamp in range(0, int(duration), interval)
    ][:maximum]
    if not timestamps:
        return []

    os.makedirs(settings.temp_dir, exist_ok=True)
    folder = tempfile.mkdtemp(prefix="humaniti-frames-", dir=settings.temp_dir)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(_extract_frame, path, timestamp, folder, index)
                for index, timestamp in enumerate(timestamps)
            ]
            extracted = [future.result() for future in futures]
        return [frame for frame in extracted if frame is not None]
    finally:
        for name in os.listdir(folder):
            os.remove(os.path.join(folder, name))
        os.rmdir(folder)


def analyze(path: str, transcript: dict) -> dict:
    sampled = frames(path)
    if not sampled:
        return {
            "assessable": False,
            "score": None,
            "reason": "No video frames extracted",
            "observations": [],
            "limitations": ["Only a bounded set of representative frames is inspected."],
        }

    excerpts = []
    for timestamp, _ in sampled:
        nearby = [
            {
                "index": index,
                "start": segment.get("start"),
                "end": segment.get("end"),
                "speaker_role": segment.get("speaker_role"),
                "text": segment.get("text", ""),
            }
            for index, segment in enumerate(transcript.get("segments", []))
            if timestamp - 15 <= segment.get("start", 0) <= timestamp + 15
        ]
        excerpts.append({"timestamp_seconds": timestamp, "transcript_segments": nearby[:10]})

    image_bytes = [image_data for _, image_data in sampled]
    result = analyze_images(
        VIDEO_SCREEN_ANALYSIS_INSTRUCTIONS,
        {
            "sampled_timestamps": [timestamp for timestamp, _ in sampled],
            "transcript_context": excerpts,
        },
        image_bytes,
    )
    observations = result.get("observations", [])
    scores = [
        float(item["readability_score"])
        for item in observations
        if isinstance(item, dict)
        and isinstance(item.get("readability_score"), (int, float))
        and 0 <= item["readability_score"] <= 100
    ]
    result["score"] = round(sum(scores) / len(scores), 2) if scores else None
    result["limitations"] = ["Analysis covers sampled frames only, not continuous video."]
    return result
