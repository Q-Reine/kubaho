"""Database engine and session. PostgreSQL + TimescaleDB in Docker, SQLite for local dev/tests."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings

_sqlite = settings.database_url.startswith("sqlite")
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if _sqlite else {},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    """Naive UTC timestamp (all datetimes are stored as UTC)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def init_db() -> None:
    from . import models  # noqa: F401  (registers tables)

    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
                conn.execute(text("SELECT create_hypertable('motion_minutes', 'ts', if_not_exists => TRUE, migrate_data => TRUE)"))
            except Exception as exc:  # plain PostgreSQL without TimescaleDB still works
                print(f"[db] TimescaleDB not enabled: {exc}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
