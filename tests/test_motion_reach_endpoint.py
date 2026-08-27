import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

PROPER_ARM_BODY = {
    "name": "reach_test",
    "links": [
        {"name": "upper_arm", "length_m": 0.3, "mass_kg": 1.5},
        {"name": "forearm", "length_m": 0.25, "mass_kg": 1.0},
    ],
    "joints": [
        {"name": "shoulder", "joint_type": "revolute", "parent_link": "base_link",
         "child_link": "upper_arm", "axis": [0, 1, 0], "lower_limit_rad": -1.57,
         "upper_limit_rad": 1.57, "max_torque_nm": 15.0},
        {"name": "elbow", "joint_type": "revolute", "parent_link": "upper_arm",
         "child_link": "forearm", "axis": [0, 1, 0], "lower_limit_rad": -2.5,
         "upper_limit_rad": 2.5, "max_torque_nm": 8.0},
    ],
    "payload_mass_kg": 0.5,
}


def test_plan_reach_endpoint_succeeds_for_reachable_target(auth_headers):
    response = client.post(
        "/motion/reach?target_x=0.4&target_y=0.0&target_z=0.85",
        json=PROPER_ARM_BODY,
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ik_result"]["reachable"] is True
    assert data["overall_success"] is True
    assert "trajectory" in data


def test_plan_reach_endpoint_returns_400_without_payload(auth_headers):
    body_no_payload = dict(PROPER_ARM_BODY)
    body_no_payload["payload_mass_kg"] = 0.0

    response = client.post(
        "/motion/reach?target_x=0.4&target_y=0.0&target_z=0.85",
        json=body_no_payload,
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert "payload_mass_kg" in response.json()["detail"]


def test_plan_reach_endpoint_can_skip_trajectory_recording(auth_headers):
    response = client.post(
        "/motion/reach?target_x=0.4&target_y=0.0&target_z=0.85&record_trajectory=false",
        json=PROPER_ARM_BODY,
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "trajectory" not in data
