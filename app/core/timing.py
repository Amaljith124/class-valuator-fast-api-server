import logging
import time
from contextlib import asynccontextmanager


@asynccontextmanager
async def timed_stage(logger: logging.Logger, stage: str, evaluation_id: str | None):
    started_at = time.perf_counter()
    try:
        yield
    finally:
        logger.info(
            "[pipeline.timing] stage=%s evaluation_id=%s seconds=%.2f",
            stage,
            evaluation_id or "none",
            time.perf_counter() - started_at,
        )
