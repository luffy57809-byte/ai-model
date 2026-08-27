import io
import trimesh
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.storage import mesh_store
from src.storage import database

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_mesh_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(mesh_store, "MESH_STORAGE_DIR", tmp_path / "meshes")
    from src.urdf_generator import generator
    monkeypatch.setattr(generator, "MESH_STORAGE_DIR", tmp_path / "meshes")

    test_db_path = tmp_path / "test.db"
    test_engine = database.create_engine(
        f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}
    )
    test_session_local = database.sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
    database.Base.metadata.create_all(bind=test_engine)
    monkeypatch.setattr(mesh_store, "SessionLocal", test_session_local)

    yield


def test_uploaded_mesh_file_can_be_downloaded_back(auth_headers):
    box = trimesh.creation.box(extents=[0.03, 0.03, 0.3])
    stl_bytes = box.export(file_type="stl")

    upload_response = client.post(
        "/meshes/upload",
        files={"file": ("test.stl", stl_bytes, "application/octet-stream")},
        data={"target_mass_kg": "1.0"},
        headers=auth_headers,
    )
    mesh_id = upload_response.json()["mesh_id"]

    file_response = client.get(f"/meshes/{mesh_id}/file", headers=auth_headers)
    assert file_response.status_code == 200

    # Not byte-identical: mesh_processor.py normalizes the mesh origin
    # (shifts it so local Z=0 is the proximal/mounting face) before
    # storing it, so the downloaded file is a re-exported, translated
    # version of the upload - same shape and volume, different origin.
    original = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")
    downloaded = trimesh.load(io.BytesIO(file_response.content), file_type="stl", force="mesh")

    assert downloaded.volume == pytest.approx(original.volume, rel=1e-6)
    assert len(downloaded.vertices) == len(original.vertices)
    # The normalized mesh's bounding box should start at Z=0.
    assert downloaded.bounds[0][2] == pytest.approx(0.0, abs=1e-6)


def test_nonexistent_mesh_file_returns_404(auth_headers):
    response = client.get("/meshes/does_not_exist/file", headers=auth_headers)
    assert response.status_code == 404


def test_path_traversal_in_mesh_id_is_sanitized(auth_headers):
    response = client.get("/meshes/..%2F..%2F..%2Fetc%2Fpasswd/file", headers=auth_headers)
    assert response.status_code == 404
