import json
import logging
import random
import time

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)
_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=settings.gemini_timeout_ms),
        )
    return _client


def _parse_json_object(gemini_response_text: str) -> dict:
    try:
        parsed_value = json.loads(gemini_response_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini returned invalid JSON") from exc
    if not isinstance(parsed_value, dict):
        raise ValueError("Gemini response must be a JSON object")
    return parsed_value


def _thinking_config(thinking: bool | None, model_name: str):
    if thinking is not False:
        return None
    model_name = model_name.lower()
    if "gemini-3" in model_name:
        return types.ThinkingConfig(thinking_level="minimal")
    return types.ThinkingConfig(thinking_budget=0)


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    response = getattr(error, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    try:
        return int(status)
    except (TypeError, ValueError):
        return None


def _is_timeout(error: Exception) -> bool:
    return isinstance(error, (TimeoutError,)) or "timeout" in error.__class__.__name__.lower()


def _is_retryable(error: Exception) -> bool:
    return _status_code(error) in {429, 500, 503} or _is_timeout(error)


def _generate_with_retry(contents, instructions: str, thinking: bool | None, model_name: str):
    for attempt in range(3):
        request_thinking = thinking
        while True:
            try:
                config_values = {
                    "system_instruction": instructions,
                    "response_mime_type": "application/json",
                    "temperature": 0,
                    "max_output_tokens": 8192,
                }
                thinking_config = _thinking_config(request_thinking, model_name)
                if thinking_config is not None:
                    config_values["thinking_config"] = thinking_config
                return client().models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_values),
                )
            except Exception as error:
                rejected_thinking = (
                    request_thinking is False
                    and _status_code(error) == 400
                    and "thinking" in str(error).lower()
                )
                if rejected_thinking:
                    logger.warning("[gemini.generate] model rejected thinking configuration; retrying without it")
                    request_thinking = None
                    continue
                if _is_retryable(error) and attempt < 2:
                    delay = (0.5 * (2**attempt)) + random.uniform(0, 0.25)
                    logger.warning(
                        "[gemini.generate] transient request failure status=%s retry=%s delay_seconds=%.2f",
                        _status_code(error),
                        attempt + 1,
                        delay,
                    )
                    time.sleep(delay)
                    break
                raise
    raise RuntimeError("Gemini request exhausted retry attempts")


def _response_json(response) -> dict:
    response_text = response.text
    if not response_text:
        raise ValueError("Gemini response contained no output text")
    return _parse_json_object(response_text)


def generate_json(
    instructions: str,
    payload: dict,
    thinking: bool | None = None,
    model_name: str | None = None,
) -> dict:
    return _response_json(
        _generate_with_retry(
            json.dumps(payload, ensure_ascii=False),
            instructions,
            thinking,
            model_name or settings.gemini_model,
        )
    )


def analyze_images(
    instructions: str,
    text_payload: dict,
    image_bytes: list[bytes],
    thinking: bool | None = None,
) -> dict:
    contents: list = [json.dumps(text_payload, ensure_ascii=False)]
    contents.extend(types.Part.from_bytes(data=data, mime_type="image/jpeg") for data in image_bytes)
    return _response_json(
        _generate_with_retry(contents, instructions, thinking, settings.gemini_model)
    )
