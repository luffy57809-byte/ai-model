"""
Storage for uploaded mesh files, per-user scoped. Binary mesh bytes
stay on disk (one .stl file per mesh, under MESH_STORAGE_DIR) - in
production this path points to a Render persistent disk mount; locally
it's just data/meshes/ as before. Computed mass properties live in the
database, scoped by user_id so a user can only load/delete their OWN
uploaded meshes' properties.
"""

import json
import uuid
from pathlib import Path

from src.storage.database import SessionLocal, MeshRecord

MESH_STORAGE_DIR = Path("data/meshes")


def save_mesh_file(mesh_bytes: bytes, user_id: str, properties: dict | None = None) -> str:
    MESH_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    mesh_id = uuid.uuid4().hex
    path = MESH_STORAGE_DIR / f"{mesh_id}.stl"
    path.write_bytes(mesh_bytes)

    if properties is not None:
        session = SessionLocal()
        try:
            session.add(MeshRecord(mesh_id=mesh_id, user_id=user_id, properties_json=json.dumps(properties)))
            session.commit()
        finally:
            session.close()

    return mesh_id


def load_mesh_properties(mesh_id: str, user_id: str) -> dict:
    session = SessionLocal()
    try:
        row = session.get(MeshRecord, mesh_id)
        if row is None or row.user_id != user_id:
            raise FileNotFoundError(
                f"No stored properties for mesh_id '{mesh_id}' - it may have been "
                f"uploaded before property persistence was added, never uploaded, "
                f"or belongs to a different user."
            )
        return json.loads(row.properties_json)
    finally:
        session.close()


def mesh_file_exists(mesh_id: str) -> bool:
    return (MESH_STORAGE_DIR / f"{mesh_id}.stl").exists()


def delete_mesh_file(mesh_id: str, user_id: str) -> bool:
    session = SessionLocal()
    try:
        row = session.get(MeshRecord, mesh_id)
        if row is None or row.user_id != user_id:
            return False

        path = MESH_STORAGE_DIR / f"{mesh_id}.stl"
        if path.exists():
            path.unlink()

        session.delete(row)
        session.commit()
        return True
    finally:
        session.close()
