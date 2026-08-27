from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_list_motors_endpoint_returns_full_library(auth_headers):
    response = client.get("/components/motors", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 4
    ids = {m["id"] for m in data}
    assert ids == {"sg90", "mg996r", "dynamixel_xl430", "dynamixel_xm430"}


def test_list_motors_endpoint_returns_usable_torque_and_velocity(auth_headers):
    response = client.get("/components/motors", headers=auth_headers)
    data = response.json()
    for motor in data:
        assert motor["stall_torque_nm"] > 0
        assert motor["max_velocity_rad_s"] > 0
