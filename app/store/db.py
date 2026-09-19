from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import AppConfig
from app.store.models import Base

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Store:
    def __init__(self, db_path: str | Path, owner_id: str = "local"):
        self.owner_id = owner_id
        self.db_path = Path(db_path)
        if not self.db_path.is_absolute():
            self.db_path = PROJECT_ROOT / self.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        url = "sqlite+aiosqlite:///" + self.db_path.resolve().as_posix()
        self.engine: AsyncEngine = create_async_engine(url, future=True)
        self.session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine, expire_on_commit=False
        )

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "Store":
        return cls(cfg.db_path, owner_id=cfg.owner_id)

    async def create_tables(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()

    def session(self) -> AsyncSession:
        return self.session_factory()


_store: Optional[Store] = None


def get_store() -> Store:
    if _store is None:
        raise RuntimeError("store is not initialized")
    return _store


def set_store(store: Store) -> None:
    global _store
    _store = store
