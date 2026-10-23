import pytest

from app.services.transcript_cleaning_service import clean_transcript, remove_loops


def _transcript(*texts, language="en"):
    segments = [
        {"start": float(index), "end": float(index + 1), "text": text, "speaker_role": "unknown"}
        for index, text in enumerate(texts)
    ]
    return {"language": language, "duration": 100.0, "segments": segments, "text": " ".join(texts)}


def test_a_long_repeated_phrase_is_removed_but_surrounding_speech_is_kept():
    text = "Okay so today we learn zugzwang. " + "I'll be more careful " * 60 + "Now look at the board."

    cleaned, loops = remove_loops(text)

    assert cleaned == "Okay so today we learn zugzwang. Now look at the board."
    assert loops == 1


@pytest.mark.parametrize(
    "text",
    ["No! " * 100, "yes, " * 80, "King 2, 3, " * 40, "It was a good game. " * 30, "Hm " * 50],
)
def test_pure_loops_leave_nothing(text):
    assert remove_loops(text)[0] == ""


@pytest.mark.parametrize("text", ["Yes yes yes", "Go go go now", "one two one two", "Rook to h1. Rook to h1."])
def test_ordinary_repetition_is_kept(text):
    assert remove_loops(text) == (text, 0)


def test_sound_tags_and_music_are_removed():
    result = clean_transcript(_transcript("(buzzing)", "[BLANK_AUDIO]", "Hello everyone (clicking) welcome", "♪♪"))

    assert [segment["text"] for segment in result["segments"]] == ["Hello everyone welcome"]
    assert result["cleaning"]["tags_removed"] == 4
    assert result["cleaning"]["empty_removed"] == 3


def test_stray_script_hallucinations_are_dropped():
    english = ["We start with the concept of opposition today."] * 1 + [
        f"Now look at example number {index} on the board carefully." for index in range(20)
    ]
    result = clean_transcript(_transcript(*english, "ஏன் ஏன் வேறு நம்பித்து", "यह लग़ा यह लग़ा"))

    assert result["cleaning"]["stray_script_removed"] == 2
    assert all("ஏன்" not in segment["text"] for segment in result["segments"])


def test_a_class_taught_in_another_script_is_not_treated_as_stray():
    malayalam = ["ഇന്ന് നമ്മൾ ഒരു പുതിയ ആശയം പഠിക്കാം"] * 1 + [f"ഇത് ഉദാഹരണം {index} ആണ്" for index in range(10)]
    result = clean_transcript(_transcript(*malayalam, "Rook to h1 is the answer", language="ml"))

    assert result["cleaning"]["stray_script_removed"] == 0
    assert len(result["segments"]) == 12


def test_repeated_identical_segments_are_collapsed():
    result = clean_transcript(_transcript("Okay.", "That was a good game.", "That was a good game.", "That was a good game.", "That was a good game."))

    texts = [segment["text"] for segment in result["segments"]]
    assert texts.count("That was a good game.") == 2
    assert result["cleaning"]["duplicates_removed"] == 2


def test_text_is_rebuilt_and_the_input_is_not_modified():
    original = _transcript("Hello there everyone", "(buzzing)")
    snapshot = [dict(segment) for segment in original["segments"]]

    result = clean_transcript(original)

    assert result["text"] == "Hello there everyone"
    assert original["segments"] == snapshot
    assert result["cleaning"]["segments_in"] == 2 and result["cleaning"]["segments_out"] == 1


@pytest.mark.parametrize("transcript", [{}, {"segments": None}, {"segments": []}, {"segments": [{"text": None}]}])
def test_empty_or_odd_transcripts_never_raise(transcript):
    result = clean_transcript(transcript)

    assert result["segments"] == []
    assert result["text"] == ""
