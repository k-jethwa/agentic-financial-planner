import pytest

from equity_research.storage.database import create_sqlite_engine, init_db, session_factory


@pytest.fixture
def session():
    engine = create_sqlite_engine(":memory:")
    init_db(engine)
    make_session = session_factory(engine)
    with make_session() as session:
        yield session
