import json
import logging
import random
import re
import time

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)
_client: genai.Client | None = None

# Thinking tokens count toward the output budget, so this must be generous.
MAX_OUTPUT_TOKENS = 32768
SAFE_OUTPUT_TOKENS = 8192  # accepted by every model; used if a model rejects the larger limit
MAX_REQUEST_ATTEMPTS = 4
_CODE_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


class GeminiOutputError(ValueError):
    """The model answered, but the answer cannot be used (empty, invalid JSON, wrong shape)."""


class GeminiOutputTruncated(GeminiOutputError):
    """The model ran out of output tokens; asking again would truncate again."""


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=settings.gemini_timeout_ms),
        )
    return _client


def _parse_json_object(gemini_response_text: str, list_key: str | None = None) -> dict:
    text = _CODE_FENCE.sub(r"\1", gemini_response_text)
    try:
        parsed_value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GeminiOutputError("Gemini returned invalid JSON") from exc
    if list_key and isinstance(parsed_value, list):
        # The model sometimes returns the list directly instead of {list_key: [...]}.
        parsed_value = {list_key: parsed_value}
    if not isinstance(parsed_value, dict):
        raise GeminiOutputError("Gemini response must be a JSON object")
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


def _is_transient_network_error(error: Exception) -> bool:
    name = error.__class__.__name__.lower()
    return isinstance(error, TimeoutError) or any(
        marker in name for marker in ("timeout", "connecterror", "remoteprotocol", "readerror")
    )


def _is_retryable(error: Exception) -> bool:
    return _status_code(error) in {408, 429, 500, 502, 503, 504} or _is_transient_network_error(error)


def _retry_delay_seconds(error: Exception, retries_done: int) -> float:
    # Rate-limit windows are per minute, so back off much longer for 429 than for a blip.
    base = 5.0 if _status_code(error) == 429 else 1.0
    return base * (2 ** (retries_done - 1)) + random.uniform(0, 0.5)


def _generate_once(
    contents,
    instructions: str,
    thinking: bool | None,
    model_name: str,
    max_output_tokens: int,
    schema: dict | None = None,
):
    config_values = {
        "system_instruction": instructions,
        "response_mime_type": "application/json",
        "temperature": 0,
        "max_output_tokens": max_output_tokens,
    }
    if schema is not None:
        config_values["response_json_schema"] = schema
    thinking_config = _thinking_config(thinking, model_name)
    if thinking_config is not None:
        config_values["thinking_config"] = thinking_config
    return client().models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(**config_values),
    )


def _generate_with_retry(
    contents,
    instructions: str,
    thinking: bool | None,
    model_name: str,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    schema: dict | None = None,
):
    request_schema = schema
    request_thinking = thinking
    request_max_tokens = max_output_tokens
    retries_done = 0
    while True:
        try:
            return _generate_once(
                contents, instructions, request_thinking, model_name, request_max_tokens, request_schema
            )
        except Exception as error:
            message = str(error).lower()
            rejected_output_limit = (
                _status_code(error) == 400
                and request_max_tokens > SAFE_OUTPUT_TOKENS
                and ("max_output_tokens" in message or "maxoutputtokens" in message or "output token" in message)
            )
            if rejected_output_limit:
                logger.warning(
                    "[gemini.generate] model rejected output limit %s; retrying with %s",
                    request_max_tokens,
                    SAFE_OUTPUT_TOKENS,
                )
                request_max_tokens = SAFE_OUTPUT_TOKENS
                continue
            rejected_schema = (
                request_schema is not None
                and _status_code(error) == 400
                and "schema" in message
            )
            if rejected_schema:
                # Prompt-described shape plus tolerant parsing still apply without the schema.
                logger.warning("[gemini.generate] model rejected response schema; retrying without it: %s", error)
                request_schema = None
                continue
            rejected_thinking = (
                request_thinking is False
                and _status_code(error) == 400
                and "thinking" in message
            )
            if rejected_thinking:
                logger.warning("[gemini.generate] model rejected thinking configuration; retrying without it")
                request_thinking = None
                continue
            retries_done += 1
            if _is_retryable(error) and retries_done < MAX_REQUEST_ATTEMPTS:
                delay = _retry_delay_seconds(error, retries_done)
                logger.warning(
                    "[gemini.generate] transient request failure status=%s model=%s retry=%s delay_seconds=%.2f",
                    _status_code(error),
                    model_name,
                    retries_done,
                    delay,
                )
                time.sleep(delay)
                continue
            raise


def _finish_reason(response) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    reason = getattr(candidates[0], "finish_reason", None)
    return str(getattr(reason, "name", reason) or "")


def _response_json(response, list_key: str | None = None) -> dict:
    response_text = response.text
    if _finish_reason(response).upper().endswith("MAX_TOKENS"):
        raise GeminiOutputTruncated("Gemini output was truncated (MAX_TOKENS)")
    if not response_text:
        raise GeminiOutputError("Gemini response contained no output text")
    return _parse_json_object(response_text, list_key)


def _generate_json_response(
    contents,
    instructions: str,
    thinking: bool | None,
    model_name: str,
    list_key: str | None,
    max_output_tokens: int | None,
    schema: dict | None = None,
) -> dict:
    """Call the model and parse its answer, asking once more if the answer is unusable."""
    last_error: GeminiOutputError | None = None
    for attempt in range(2):
        response = _generate_with_retry(
            contents,
            instructions,
            thinking,
            model_name,
            max_output_tokens or MAX_OUTPUT_TOKENS,
            schema,
        )
        try:
            return _response_json(response, list_key)
        except GeminiOutputTruncated:
            raise
        except GeminiOutputError as error:
            last_error = error
            logger.warning(
                "[gemini.generate] unusable model output model=%s attempt=%s reason=%s",
                model_name,
                attempt + 1,
                error,
            )
    raise last_error


def generate_json(
    instructions: str,
    payload: dict,
    thinking: bool | None = None,
    model_name: str | None = None,
    list_key: str | None = None,
    max_output_tokens: int | None = None,
    schema: dict | None = None,
) -> dict:
    return _generate_json_response(
        json.dumps(payload, ensure_ascii=False),
        instructions,
        thinking,
        model_name or settings.gemini_model,
        list_key,
        max_output_tokens,
        schema,
    )


def analyze_images(
    instructions: str,
    text_payload: dict,
    image_bytes: list[bytes],
    thinking: bool | None = None,
    list_key: str | None = None,
    max_output_tokens: int | None = None,
    schema: dict | None = None,
) -> dict:
    contents: list = [json.dumps(text_payload, ensure_ascii=False)]
    contents.extend(types.Part.from_bytes(data=data, mime_type="image/jpeg") for data in image_bytes)
    return _generate_json_response(
        contents,
        instructions,
        thinking,
        settings.gemini_model,
        list_key,
        max_output_tokens,
        schema,
    )
