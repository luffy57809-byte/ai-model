"""
Database-backed, per-user persistence for arm designs.

Only the ArmConfig (the design) is saved - not analysis results. Torque
checks and lift tests are always recomputed live from the current code,
so a saved design never goes stale relative to bug fixes or model
changes in the analysis itself.

SECURITY NOTE: config.name is user-supplied and becomes part of the
slug (composite primary key with user_id). _slugify() strips everything
except alphanumerics, underscore, and hyphen.

PER-USER SCOPING: every function now takes a user_id, and all queries
filter by it - a user can only see, load, or delete their OWN saved
designs. slug alone is no longer globally unique (two users can each
have a design named "my_arm"); the real identity is (slug, user_id).
"""

import datetime
import json
import re

from src.storage.database import SessionLocal, SavedDesign
from src.urdf_generator.schema import ArmConfig


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip())
    slug = slug.strip("_") or "unnamed"
    return slug[:100]


def save_design(config: ArmConfig, user_id: str) -> dict:
    slug = _slugify(config.name)
    saved_at = datetime.datetime.now(datetime.timezone.utc)

    session = SessionLocal()
    try:
        existing = session.get(SavedDesign, (slug, user_id))
        if existing:
            existing.name = config.name
            existing.config_json = config.model_dump_json()
            existing.saved_at = saved_at
        else:
            session.add(SavedDesign(
                slug=slug, user_id=user_id, name=config.name,
                config_json=config.model_dump_json(), saved_at=saved_at,
            ))
        session.commit()
    finally:
        session.close()

    return {"name": config.name, "slug": slug, "saved_at": saved_at.isoformat()}


def list_designs(user_id: str) -> list[dict]:
    session = SessionLocal()
    try:
        rows = (
            session.query(SavedDesign)
            .filter(SavedDesign.user_id == user_id)
            .order_by(SavedDesign.saved_at.desc())
            .all()
        )
        return [
            {"name": row.name, "slug": row.slug, "saved_at": row.saved_at.isoformat() if row.saved_at else None}
            for row in rows
        ]
    finally:
        session.close()


def load_design(slug: str, user_id: str) -> ArmConfig:
    safe_slug = _slugify(slug)
    session = SessionLocal()
    try:
        row = session.get(SavedDesign, (safe_slug, user_id))
        if row is None:
            raise FileNotFoundError(f"No saved design found for '{slug}'")
        return ArmConfig(**json.loads(row.config_json))
    finally:
        session.close()


def delete_design(slug: str, user_id: str) -> bool:
    safe_slug = _slugify(slug)
    session = SessionLocal()
    try:
        row = session.get(SavedDesign, (safe_slug, user_id))
        if row is None:
            return False
        session.delete(row)
        session.commit()
        return True
    finally:
        session.close()
