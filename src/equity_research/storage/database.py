"""SQLite persistence layer: engine setup and ORM tables.

`research_runs` holds one row per run with its request, status, and final
report. `trace_events` is append-only: nodes only ever add a row, they
never update or delete one, so the trace viewer always sees the complete
history of a run.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class ResearchRunORM(Base):
    __tablename__ = "research_runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10))
    question: Mapped[str] = mapped_column(Text)
    report_mode: Mapped[str] = mapped_column(String(16))
    as_of_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    request_json: Mapped[str] = mapped_column(Text)
    report_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String(32))
    updated_at: Mapped[str] = mapped_column(String(32))


class TraceEventORM(Base):
    __tablename__ = "trace_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("research_runs.run_id"), index=True)
    node: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    occurred_at: Mapped[str] = mapped_column(String(32))
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)


def create_sqlite_engine(db_path: str | Path = ":memory:", *, echo: bool = False):
    """Create a SQLite engine. `:memory:` is used throughout the test suite.

    A plain `sqlite:///:memory:` engine hands out a *fresh, empty* database
    on every new connection by default -- fine for a single long-lived
    `session` fixture, but silently loses all data the moment a second
    connection is opened (e.g. one request-scoped session per API call).
    `StaticPool` pins every checkout to the same single connection so an
    in-memory engine behaves like one shared database for the process.
    """
    if str(db_path) == ":memory:":
        return create_engine(
            "sqlite:///:memory:",
            echo=echo,
            future=True,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path}", echo=echo, future=True)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


def session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
