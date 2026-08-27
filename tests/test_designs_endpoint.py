import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.storage import config_store
from src.storage import database

client = TestClient(app)

SAMPLE_BODY = {
    "name": "api_test_arm",
    "links": [{"name": "link1", "length_m": 0.2, "mass_kg": 1.0}],
    "joints": [{"name": "joint1", "joint_type": "revolute", "parent_link": "base_link",
                "child_link": "link1", "lower_limit_rad": -1.57, "upper_limit_rad": 1.57,
                "max_torque_nm": 10.0}],
    "payload_mass_kg": 0.0,
}


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Isolates each test to its own temporary SQLite database file -
    replaces the old file-path monkeypatch."""
    test_db_path = tmp_path / "test.db"
    test_engine = database.create_engine(
        f"sqlite:///{test_db_path}", connect_args={"check_same_thread": False}
    )
    test_session_local = database.sessionmaker(bind=test_engine, autoflush=False, autocommit=False)
    database.Base.metadata.create_all(bind=test_engine)

    monkeypatch.setattr(config_store, "SessionLocal", test_session_local)
    yield


def test_save_list_load_delete_round_trip(auth_headers):
    save_response = client.post("/designs", json=SAMPLE_BODY, headers=auth_headers)
    assert save_response.status_code == 200
    slug = save_response.json()["slug"]
    list_response = client.get("/designs", headers=auth_headers)
    names = {d["name"] for d in list_response.json()}
    assert "api_test_arm" in names
    get_response = client.get(f"/designs/{slug}", headers=auth_headers)
    assert get_response.json()["name"] == "api_test_arm"
    delete_response = client.delete(f"/designs/{slug}", headers=auth_headers)
    assert delete_response.status_code == 200
    assert client.get(f"/designs/{slug}", headers=auth_headers).status_code == 404


def test_save_rejects_invalid_config(auth_headers):
    broken_body = dict(SAMPLE_BODY)
    broken_body["joints"] = [{"name": "joint1", "joint_type": "revolute",
                               "parent_link": "base_link", "child_link": "nonexistent_link",
                               "max_torque_nm": 10.0}]
    response = client.post("/designs", json=broken_body, headers=auth_headers)
    assert response.status_code == 400


def test_get_nonexistent_design_returns_404(auth_headers):
    assert client.get("/designs/definitely_not_saved", headers=auth_headers).status_code == 404


def test_delete_nonexistent_design_returns_404(auth_headers):
    assert client.delete("/designs/definitely_not_saved", headers=auth_headers).status_code == 404
