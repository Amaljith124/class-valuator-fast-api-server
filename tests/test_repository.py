import json

import numpy as np

from app.database import repository


def test_dumps_replaces_non_finite_numbers_and_strips_nul_characters():
    payload = {"a": float("nan"), "b": float("inf"), "c": "te\x00xt", "d": [1.5, float("-inf")], "e": {"f": "\x00"}}

    loaded = json.loads(repository._dumps(payload))

    assert loaded == {"a": None, "b": None, "c": "text", "d": [1.5, None], "e": {"f": ""}}


def test_dumps_handles_numpy_values():
    payload = {"score": np.float32(1.5), "count": np.int64(3), "flag": np.bool_(True)}

    loaded = json.loads(repository._dumps(payload))

    assert loaded == {"score": 1.5, "count": 3, "flag": True}


def test_dumps_output_is_valid_strict_json():
    json.loads(repository._dumps({"x": float("nan")}), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def test_clean_text_strips_nul_and_handles_none():
    assert repository._clean_text("a\x00b") == "ab"
    assert repository._clean_text(None) == ""
