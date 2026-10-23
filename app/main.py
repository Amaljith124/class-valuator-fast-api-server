import logging
import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from app.core.config import settings
from app.database.db import connect_db, close_db
from app.api.routes import router
from app.services.transcription_service import warm_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_db(settings.database_url)
    try:
        await asyncio.to_thread(_clean_old_temp_files)
        await asyncio.to_thread(warm_model)
        yield
    finally:
        await close_db()


def _clean_old_temp_files() -> None:
    temp_dir = Path(settings.temp_dir)
    if not temp_dir.exists():
        return
    cutoff = time.time() - 3600
    removed = 0
    for path in temp_dir.iterdir():
        if (
            path.name.startswith("tmp")
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_mtime < cutoff
        ):
            path.unlink()
            removed += 1
    logging.getLogger(__name__).info("[startup.cleanup] removed_old_temp_files=%s", removed)

app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.include_router(router)
