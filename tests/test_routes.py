import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app
from app.services.pipeline_manager import EvaluationAlreadyCompleted

HEADERS = {"x-internal-token": "test-internal-token"}
BODY = {"object_key": "recordings/2026-10-23/a.mp4", "evaluation_id": "abc"}


@pytest.fixture
def client():
    # Not used as a context manager, so the lifespan (database, Whisper warm-up) does not start.
    return TestClient(app, raise_server_exceptions=False)


def _fake_run(outcome):
    async def run(object_key, evaluation_id):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return run


def test_analyze_requires_the_internal_token(client):
    assert client.post("/api/v1/analyze", json=BODY).status_code == 401
    assert client.post("/api/v1/analyze", json=BODY, headers={"x-internal-token": "wrong"}).status_code == 401


def test_analyze_success(client, monkeypatch):
    monkeypatch.setattr(routes, "run", _fake_run({"status": "completed"}))

    response = client.post("/api/v1/analyze", json=BODY, headers=HEADERS)

    assert response.status_code == 200 and response.json()["success"] is True


def test_already_completed_is_a_409(client, monkeypatch):
    monkeypatch.setattr(routes, "run", _fake_run(EvaluationAlreadyCompleted("Evaluation is already completed")))

    assert client.post("/api/v1/analyze", json=BODY, headers=HEADERS).status_code == 409


def test_missing_evaluation_is_a_404(client, monkeypatch):
    monkeypatch.setattr(routes, "run", _fake_run(LookupError("Evaluation not found")))

    assert client.post("/api/v1/analyze", json=BODY, headers=HEADERS).status_code == 404


def test_model_output_errors_are_a_500_not_a_409(client, monkeypatch):
    """Regression: any ValueError used to be reported as 409 Conflict."""
    monkeypatch.setattr(routes, "run", _fake_run(ValueError("Gemini response must be a JSON object")))

    assert client.post("/api/v1/analyze", json=BODY, headers=HEADERS).status_code == 500


@pytest.mark.parametrize("key", ["", "/absolute", "recordings/../secret", "https://evil/x"])
def test_unsafe_object_keys_are_rejected(client, key):
    response = client.post("/api/v1/analyze", json={"object_key": key}, headers=HEADERS)

    assert response.status_code == 422
