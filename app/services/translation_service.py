from app.services.gemini_service import generate_json
from app.templates.prompt_templates import TRANSCRIPT_TRANSLATION_INSTRUCTIONS


def translate_to_english(transcript: dict) -> tuple[dict, bool]:
    """Return an English copy of a transcript and whether translation was needed."""
    language = str(transcript.get("language") or "").strip().lower()
    segments = transcript.get("segments", [])
    english_transcript = {
        **transcript,
        "segments": [dict(segment) for segment in segments],
    }

    if language == "en" or language.startswith("en-"):
        return english_transcript, False

    if not segments:
        english_transcript["text"] = ""
        return english_transcript, True

    translated_by_index = {}
    batch_size = 80

    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start:batch_start + batch_size]
        indexed_segments = [
            {"index": batch_start + offset, "text": str(segment.get("text") or "")}
            for offset, segment in enumerate(batch)
        ]
        translation = generate_json(
            TRANSCRIPT_TRANSLATION_INSTRUCTIONS,
            {
                "source_language": language or "unknown",
                "target_language": "English",
                "segments": indexed_segments,
            },
        )
        translated_segments = translation.get("segments", [])

        if not isinstance(translated_segments, list):
            raise ValueError("Gemini transcript translation did not return a segment list")

        for translated_segment in translated_segments:
            if not isinstance(translated_segment, dict):
                continue
            index = translated_segment.get("index")
            translated_text = translated_segment.get("text")
            if isinstance(index, int) and isinstance(translated_text, str):
                translated_by_index[index] = translated_text

        expected_indexes = range(batch_start, batch_start + len(batch))
        missing_indexes = [index for index in expected_indexes if index not in translated_by_index]
        if missing_indexes:
            raise ValueError(
                "Gemini transcript translation omitted segment indexes: "
                + ", ".join(str(index) for index in missing_indexes)
            )

    for index, segment in enumerate(english_transcript["segments"]):
        segment["text"] = translated_by_index[index]

    english_transcript["text"] = " ".join(
        segment["text"] for segment in english_transcript["segments"] if segment["text"]
    )
    english_transcript["source_language"] = language or None
    return english_transcript, True
