import os
from typing import Optional

from dotenv import load_dotenv
import asyncpg


load_dotenv()


DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")


if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in the environment.")


pool: Optional[asyncpg.Pool] = None


async def connect_db():
    """
    Create and initialize the database connection pool.
    Should be called during application startup.
    """
    global pool
    pool = await asyncpg.create_pool(
        DATABASE_URL, min_size=1, max_size=10, statement_cache_size=0
    )
    print("Connected to the database 🚀")


async def close_db():
    """
    Close the database connection pool.
    Should be called during application shutdown.
    """
    global pool
    if pool:
        await pool.close()
        print("Database connection pool closed")


def get_pool() -> asyncpg.Pool:
    """
    Get the database connection pool.
    Raises RuntimeError if pool is not initialized.
    """
    if pool is None:
        raise RuntimeError("Database pool is not initialized. Call connect_db() first.")
    return pool
