import mimetypes
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from minio import Minio

from app.core.config import settings

_client = None


def client() -> Minio:
    global _client

    if _client is None:
        endpoint = urlsplit(settings.r2_endpoint)
        expected_hostname = f"{settings.r2_account_id}.r2.cloudflarestorage.com"
        if endpoint.scheme != "https" or endpoint.hostname != expected_hostname or endpoint.path:
            raise ValueError("R2_ENDPOINT must be an HTTPS account endpoint without a path")

        _client = Minio(
            endpoint.hostname,
            access_key=settings.r2_access_key_id,
            secret_key=settings.r2_secret_access_key,
            secure=True,
            region=settings.r2_region,
        )

    return _client


def validate_object_key(object_key: str) -> str:
    parts = object_key.split("/") if object_key else []
    if (
        not object_key
        or not object_key.startswith("recordings/")
        or object_key.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
        or "://" in object_key
    ):
        raise ValueError("Invalid R2 object key")

    return object_key


def download_to_file(object_key: str, destination_path: str) -> None:
    client().fget_object(
        settings.r2_bucket_name,
        validate_object_key(object_key),
        destination_path,
    )


def upload_file(object_key: str, source_path: str, content_type: str | None = None) -> None:
    validated_key = validate_object_key(object_key)
    detected_type = content_type or mimetypes.guess_type(source_path)[0] or "application/octet-stream"
    client().fput_object(
        settings.r2_bucket_name,
        validated_key,
        source_path,
        content_type=detected_type,
    )


def get_object_stream(object_key: str):
    return client().get_object(settings.r2_bucket_name, validate_object_key(object_key))


def object_exists(object_key: str) -> bool:
    client().stat_object(settings.r2_bucket_name, validate_object_key(object_key))
    return True


def get_object_metadata(object_key: str):
    return client().stat_object(settings.r2_bucket_name, validate_object_key(object_key))


def delete_object(object_key: str) -> None:
    client().remove_object(settings.r2_bucket_name, validate_object_key(object_key))


@contextmanager
def downloaded_object_file(object_key: str):
    validated_key = validate_object_key(object_key)
    suffix = Path(validated_key).suffix or ".bin"
    os.makedirs(settings.temp_dir, exist_ok=True)
    file_descriptor, file_path = tempfile.mkstemp(suffix=suffix, dir=settings.temp_dir)
    os.close(file_descriptor)

    try:
        download_to_file(validated_key, file_path)
        yield file_path
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
