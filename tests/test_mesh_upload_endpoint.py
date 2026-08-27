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


def _test_stl_bytes():
    box = trimesh.creation.box(extents=[0.03, 0.03, 0.3])
    box.apply_translation([0, 0, 0.15])
    return box.export(file_type="stl")


def test_upload_mesh_returns_real_computed_properties(auth_headers):
    response = client.post(
        "/meshes/upload",
        files={"file": ("test.stl", _test_stl_bytes(), "application/octet-stream")},
        data={"target_mass_kg": "1.5"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mass_kg"] == pytest.approx(1.5, abs=1e-6)
    assert data["length_m"] == pytest.approx(0.3, abs=1e-3)
    assert data["com_offset_m"] == pytest.approx(0.15, abs=1e-3)
    assert "mesh_id" in data


def test_upload_bad_mesh_returns_400_not_500(auth_headers):
    response = client.post(
        "/meshes/upload",
        files={"file": ("bad.stl", b"not a real stl file", "application/octet-stream")},
        data={"target_mass_kg": "1.0"},
        headers=auth_headers,
    )
    assert response.status_code == 400


def test_uploaded_mesh_can_be_used_in_a_full_analyze_request(auth_headers):
    upload_response = client.post(
        "/meshes/upload",
        files={"file": ("test.stl", _test_stl_bytes(), "application/octet-stream")},
        data={"target_mass_kg": "1.5"},
        headers=auth_headers,
    )
    mesh_data = upload_response.json()
    arm_body = {
        "name": "mesh_arm_test",
        "links": [{
            "name": "mesh_link", "length_m": mesh_data["length_m"], "mass_kg": mesh_data["mass_kg"],
            "mesh_id": mesh_data["mesh_id"], "com_offset_m": mesh_data["com_offset_m"],
            "inertia_ixx": mesh_data["inertia_ixx"], "inertia_iyy": mesh_data["inertia_iyy"],
            "inertia_izz": mesh_data["inertia_izz"], "inertia_ixy": mesh_data["inertia_ixy"],
            "inertia_ixz": mesh_data["inertia_ixz"], "inertia_iyz": mesh_data["inertia_iyz"],
        }],
        "joints": [{
            "name": "j1", "joint_type": "revolute", "parent_link": "base_link",
            "child_link": "mesh_link", "axis": [0, 1, 0],
            "lower_limit_rad": -1.57, "upper_limit_rad": 1.57, "max_torque_nm": 10.0,
        }],
        "payload_mass_kg": 0.0,
    }
    analyze_response = client.post("/analyze/arm?include_lift_test=false", json=arm_body, headers=auth_headers)
    assert analyze_response.status_code == 200
    result = analyze_response.json()
    assert result["torque_check"][0]["required_torque_nm"] > 0
