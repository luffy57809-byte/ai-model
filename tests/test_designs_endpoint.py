import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.storage import config_store

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
    monkeypatch.setattr(config_store, "STORAGE_DIR", tmp_path / "saved_designs")
    yield


def test_save_list_load_delete_round_trip():
    save_response = client.post("/designs", json=SAMPLE_BODY)
    assert save_response.status_code == 200
    slug = save_response.json()["slug"]
    list_response = client.get("/designs")
    names = {d["name"] for d in list_response.json()}
    assert "api_test_arm" in names
    get_response = client.get(f"/designs/{slug}")
    assert get_response.json()["name"] == "api_test_arm"
    delete_response = client.delete(f"/designs/{slug}")
    assert delete_response.status_code == 200
    assert client.get(f"/designs/{slug}").status_code == 404


def test_save_rejects_invalid_config():
    broken_body = dict(SAMPLE_BODY)
    broken_body["joints"] = [{"name": "joint1", "joint_type": "revolute",
                               "parent_link": "base_link", "child_link": "nonexistent_link",
                               "max_torque_nm": 10.0}]
    response = client.post("/designs", json=broken_body)
    assert response.status_code == 400


def test_get_nonexistent_design_returns_404():
    assert client.get("/designs/definitely_not_saved").status_code == 404


def test_delete_nonexistent_design_returns_404():
    assert client.delete("/designs/definitely_not_saved").status_code == 404
