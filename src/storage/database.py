"""
SQLAlchemy database layer for structured storage (saved designs, mesh
property metadata) - replaces the old local-file-based config_store.py
and mesh_store.py's JSON sidecar approach.

DATABASE_URL env var selects the backend: unset defaults to a local
SQLite file (data/app.db) for local development/testing without needing
a real Postgres server running; in production, Render provides a real
Postgres DATABASE_URL. Same ORM code works for both since nothing
Postgres-specific is used.
"""

import os

from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///data/app.db")

# SQLite needs this connect_arg for use across FastAPI's multiple
# threads; Postgres doesn't need or accept it.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True)  # uuid4 hex
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class SavedDesign(Base):
    __tablename__ = "saved_designs"

    # NOTE: slug alone is no longer globally unique across users (two
    # users could each save a design named "my_arm") - see the composite
    # primary key below, added when per-user scoping was introduced.
    slug = Column(String(100), primary_key=True)
    user_id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    config_json = Column(Text, nullable=False)  # ArmConfig.model_dump_json()
    saved_at = Column(DateTime(timezone=True), nullable=False)


class MeshRecord(Base):
    __tablename__ = "mesh_records"

    mesh_id = Column(String(64), primary_key=True)
    user_id = Column(String(36), nullable=False, index=True)
    properties_json = Column(Text, nullable=False)  # the response_body dict, JSON-encoded


def init_db():
    """Creates tables if they don't exist. Safe to call on every app
    startup - no-op if tables already exist."""
    import os
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_session():
    """Yields a session, closes it after use - standard SQLAlchemy
    session-per-request pattern."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
