from types import SimpleNamespace

import pytest

from app.services import gemini_service as gemini


def _response(text, finish="STOP"):
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))])


class _ApiError(Exception):
    def __init__(self, code, message="error"):
        super().__init__(message)
        self.code = code


class _FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(gemini.time, "sleep", lambda seconds: recorded.append(seconds))
    return recorded


def _use(monkeypatch, outcomes):
    fake = _FakeClient(outcomes)
    monkeypatch.setattr(gemini, "client", lambda: fake)
    return fake


def test_parse_accepts_plain_and_code_fenced_json():
    assert gemini._parse_json_object('{"a": 1}') == {"a": 1}
    assert gemini._parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert gemini._parse_json_object('```\n{"a": 1}\n```') == {"a": 1}


def test_parse_wraps_a_bare_list_only_when_a_list_key_is_given():
    assert gemini._parse_json_object('[{"x": 1}]', "observations") == {"observations": [{"x": 1}]}
    with pytest.raises(ValueError, match="must be a JSON object"):
        gemini._parse_json_object('[{"x": 1}]')


@pytest.mark.parametrize("text", ["not json", "", "{", '"just a string"', "42"])
def test_parse_rejects_unusable_text(text):
    with pytest.raises(ValueError):
        gemini._parse_json_object(text)


def test_truncated_output_is_reported_clearly_and_not_asked_again(monkeypatch, sleeps):
    fake = _use(monkeypatch, [_response('{"a":', finish="MAX_TOKENS")])

    with pytest.raises(gemini.GeminiOutputTruncated):
        gemini.generate_json("instructions", {"x": 1})

    assert fake.calls == 1


def test_unusable_output_is_asked_again_once(monkeypatch, sleeps):
    fake = _use(monkeypatch, [_response("garbage"), _response('{"ok": true}')])

    assert gemini.generate_json("instructions", {}) == {"ok": True}
    assert fake.calls == 2


def test_unusable_output_twice_raises(monkeypatch, sleeps):
    fake = _use(monkeypatch, [_response("garbage"), _response("still garbage")])

    with pytest.raises(ValueError):
        gemini.generate_json("instructions", {})

    assert fake.calls == 2


def test_list_key_rescues_a_bare_array_answer(monkeypatch, sleeps):
    _use(monkeypatch, [_response('[{"index": 0}]')])

    assert gemini.generate_json("instructions", {}, list_key="segments") == {"segments": [{"index": 0}]}


def test_rate_limits_are_retried_with_long_backoff(monkeypatch, sleeps):
    fake = _use(monkeypatch, [_ApiError(429), _ApiError(429), _response('{"ok": 1}')])

    assert gemini.generate_json("instructions", {}) == {"ok": 1}
    assert fake.calls == 3
    assert sleeps[0] >= 5 and sleeps[1] >= 10


@pytest.mark.parametrize("code", [500, 503, 504])
def test_server_errors_are_retried(monkeypatch, sleeps, code):
    fake = _use(monkeypatch, [_ApiError(code), _response('{"ok": 1}')])

    assert gemini.generate_json("instructions", {}) == {"ok": 1}
    assert fake.calls == 2


def test_timeouts_are_retried(monkeypatch, sleeps):
    class ReadTimeout(Exception):
        pass

    fake = _use(monkeypatch, [ReadTimeout("slow"), _response('{"ok": 1}')])

    assert gemini.generate_json("instructions", {}) == {"ok": 1}
    assert fake.calls == 2


def test_retries_stop_after_the_attempt_limit(monkeypatch, sleeps):
    fake = _use(monkeypatch, [_ApiError(503)] * gemini.MAX_REQUEST_ATTEMPTS)

    with pytest.raises(_ApiError):
        gemini.generate_json("instructions", {})

    assert fake.calls == gemini.MAX_REQUEST_ATTEMPTS


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_client_errors_are_not_retried(monkeypatch, sleeps, code):
    fake = _use(monkeypatch, [_ApiError(code)])

    with pytest.raises(_ApiError):
        gemini.generate_json("instructions", {})

    assert fake.calls == 1 and sleeps == []


def test_a_model_that_rejects_thinking_settings_is_retried_without_them(monkeypatch, sleeps):
    fake = _use(monkeypatch, [_ApiError(400, "thinking level not supported"), _response('{"ok": 1}')])

    assert gemini.generate_json("instructions", {}, thinking=False) == {"ok": 1}
    assert fake.calls == 2 and sleeps == []


def test_output_budget_leaves_room_for_thinking_tokens():
    assert gemini.MAX_OUTPUT_TOKENS >= 16384


def test_a_model_that_rejects_the_output_limit_is_retried_with_a_safe_one(monkeypatch, sleeps):
    seen = []

    class RecordingClient(_FakeClient):
        def _generate(self, **kwargs):
            seen.append(kwargs["config"].max_output_tokens)
            return super()._generate(**kwargs)

    fake = RecordingClient([_ApiError(400, "max_output_tokens must be <= 8192"), _response('{"ok": 1}')])
    monkeypatch.setattr(gemini, "client", lambda: fake)

    assert gemini.generate_json("instructions", {}) == {"ok": 1}
    assert seen == [gemini.MAX_OUTPUT_TOKENS, gemini.SAFE_OUTPUT_TOKENS]


def test_schema_is_sent_to_the_model(monkeypatch):
    seen = []

    class RecordingClient(_FakeClient):
        def _generate(self, **kwargs):
            seen.append(kwargs["config"].response_json_schema)
            return super()._generate(**kwargs)

    schema = {"type": "object", "properties": {"ok": {"type": "integer"}}, "required": ["ok"]}
    fake = RecordingClient([_response('{"ok": 1}')])
    monkeypatch.setattr(gemini, "client", lambda: fake)

    assert gemini.generate_json("instructions", {}, schema=schema) == {"ok": 1}
    assert seen == [schema]


def test_a_rejected_schema_is_retried_without_it(monkeypatch, sleeps):
    seen = []

    class RecordingClient(_FakeClient):
        def _generate(self, **kwargs):
            seen.append(kwargs["config"].response_json_schema)
            return super()._generate(**kwargs)

    schema = {"type": "object", "properties": {}}
    fake = RecordingClient([_ApiError(400, "Invalid JSON schema in request"), _response('{"ok": 1}')])
    monkeypatch.setattr(gemini, "client", lambda: fake)

    assert gemini.generate_json("instructions", {}, schema=schema) == {"ok": 1}
    assert seen == [schema, None]


def test_temperature_and_seed_are_sent_to_the_model(monkeypatch):
    seen = []

    class RecordingClient(_FakeClient):
        def _generate(self, **kwargs):
            config = kwargs["config"]
            seen.append((config.temperature, config.seed))
            return super()._generate(**kwargs)

    fake = RecordingClient([_response('{"a": 1}'), _response('{"a": 1}')])
    monkeypatch.setattr(gemini, "client", lambda: fake)

    gemini.generate_json("instructions", {})
    gemini.generate_json("instructions", {}, temperature=0.4, seed=9)

    assert seen == [(0, None), (0.4, 9)]
