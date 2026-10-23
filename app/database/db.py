from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

_pool: AsyncConnectionPool | None = None

async def connect_db(database_url: str) -> None:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(conninfo=database_url, min_size=1, max_size=5, kwargs={"row_factory": dict_row}, open=False)
        await _pool.open()
        await _pool.wait()
    else:
        raise RuntimeError("Database pool already initialized")

async def close_db() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None

def get_pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("Database pool is not initialized")
    return _pool

async def fetch_one(sql: str, params: tuple | list | None = None):
    async with get_pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params or ())
            return await cur.fetchone()

async def execute(sql: str, params: tuple | list | None = None) -> int:
    async with get_pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params or ())
            return cur.rowcount
