import pytest

from app.services import video_screen_service as video


def test_short_videos_keep_the_configured_interval():
    assert video._sample_timestamps(100, 30, 12) == [0.0, 30.0, 60.0, 90.0]


def test_long_videos_are_sampled_across_their_whole_length():
    timestamps = video._sample_timestamps(2668, 30, 12)

    assert len(timestamps) == 12
    assert timestamps[0] == 0.0
    assert timestamps[-1] > 2000


@pytest.mark.parametrize("duration,maximum", [(0, 12), (-5, 12), (100, 0)])
def test_no_timestamps_for_empty_inputs(duration, maximum):
    assert video._sample_timestamps(duration, 30, maximum) == []


def _transcript():
    return {"segments": [{"start": 1.0, "end": 2.0, "text": "hello"}, {"start": None, "text": "bad"}, {"text": "no start"}]}


@pytest.mark.parametrize("observations", [None, "text", {"a": 1}, 5])
def test_non_list_observations_do_not_crash(monkeypatch, observations):
    monkeypatch.setattr(video, "frames", lambda path: [(0.0, b"jpeg")])
    monkeypatch.setattr(video, "analyze_images", lambda *args, **kwargs: {"observations": observations})

    result = video.analyze("movie.mp4", _transcript())

    assert result["observations"] == []
    assert result["score"] is None


def test_scores_average_only_valid_readability_values(monkeypatch):
    monkeypatch.setattr(video, "frames", lambda path: [(0.0, b"jpeg")])
    monkeypatch.setattr(
        video,
        "analyze_images",
        lambda *args, **kwargs: {
            "observations": [
                {"readability_score": 80},
                {"readability_score": 60},
                {"readability_score": 500},
                {"readability_score": None},
                "junk",
            ]
        },
    )

    assert video.analyze("movie.mp4", _transcript())["score"] == 70.0


def test_list_key_is_passed_so_a_bare_array_is_accepted(monkeypatch):
    captured = {}
    monkeypatch.setattr(video, "frames", lambda path: [(0.0, b"jpeg")])

    def fake_analyze_images(*args, **kwargs):
        captured.update(kwargs)
        return {"observations": []}

    monkeypatch.setattr(video, "analyze_images", fake_analyze_images)
    video.analyze("movie.mp4", _transcript())

    assert captured["list_key"] == "observations"


def test_no_frames_returns_not_assessable(monkeypatch):
    monkeypatch.setattr(video, "frames", lambda path: [])

    result = video.analyze("audio-only.mp3", _transcript())

    assert result["assessable"] is False
    assert result["score"] is None
