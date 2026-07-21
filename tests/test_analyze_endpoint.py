from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_analyze_arm_accepts_the_two_link_sample():
    body = {
        "name": "two_link_arm",
        "links": [
            {"name": "upper_arm", "length_m": 0.3, "radius_m": 0.03, "mass_kg": 1.5},
            {"name": "forearm", "length_m": 0.25, "radius_m": 0.025, "mass_kg": 1.0},
        ],
        "joints": [
            {"name": "shoulder", "joint_type": "revolute", "parent_link": "base_link",
             "child_link": "upper_arm", "axis": [0, 0, 1],
             "lower_limit_rad": -3.14, "upper_limit_rad": 3.14, "max_torque_nm": 15.0},
            {"name": "elbow", "joint_type": "revolute", "parent_link": "upper_arm",
             "child_link": "forearm", "axis": [0, 1, 0],
             "lower_limit_rad": -2.5, "upper_limit_rad": 2.5, "max_torque_nm": 8.0},
        ],
        "payload_mass_kg": 0.5,
    }

    response = client.post("/analyze/arm", json=body)
    assert response.status_code == 200
    data = response.json()

    elbow = next(j for j in data["torque_check"] if j["joint_name"] == "elbow")
    assert elbow["required_torque_nm"] == 2.4525
    assert "lift_test" in data
    assert "report" not in data


def test_analyze_arm_with_a_genuinely_different_design():
    body = {
        "name": "three_link_test_arm",
        "links": [
            {"name": "link1", "length_m": 0.2, "radius_m": 0.03, "mass_kg": 2.0},
            {"name": "link2", "length_m": 0.2, "radius_m": 0.03, "mass_kg": 1.5},
            {"name": "link3", "length_m": 0.15, "radius_m": 0.02, "mass_kg": 0.8},
        ],
        "joints": [
            {"name": "joint1", "joint_type": "revolute", "parent_link": "base_link",
             "child_link": "link1", "axis": [0, 1, 0],
             "lower_limit_rad": -1.57, "upper_limit_rad": 1.57, "max_torque_nm": 20.0},
            {"name": "joint2", "joint_type": "revolute", "parent_link": "link1",
             "child_link": "link2", "axis": [0, 1, 0],
             "lower_limit_rad": -1.57, "upper_limit_rad": 1.57, "max_torque_nm": 10.0},
            {"name": "joint3", "joint_type": "revolute", "parent_link": "link2",
             "child_link": "link3", "axis": [0, 1, 0],
             "lower_limit_rad": -1.57, "upper_limit_rad": 1.57, "max_torque_nm": 0.3},
        ],
        "payload_mass_kg": 1.0,
    }

    response = client.post("/analyze/arm", json=body)
    assert response.status_code == 200
    data = response.json()

    assert len(data["torque_check"]) == 3
    joint3 = next(j for j in data["torque_check"] if j["joint_name"] == "joint3")
    assert joint3["required_torque_nm"] == 2.0601
    assert joint3["passes"] is False

    lift_joint3 = next(j for j in data["lift_test"]["joint_results"] if j["joint_name"] == "joint3")
    assert lift_joint3["passes"] is False
    assert lift_joint3["sag_deg"] > 5.0


def test_analyze_arm_rejects_invalid_config_with_clear_error():
    body = {
        "name": "broken_arm",
        "links": [
            {"name": "link1", "length_m": 0.2, "mass_kg": 1.0},
        ],
        "joints": [
            {"name": "joint1", "joint_type": "revolute", "parent_link": "base_link",
             "child_link": "nonexistent_link", "max_torque_nm": 10.0},
        ],
        "payload_mass_kg": 0.0,
    }

    response = client.post("/analyze/arm", json=body)
    assert response.status_code == 400
    assert "nonexistent_link" in str(response.json())


def test_analyze_arm_can_skip_lift_test_for_speed():
    body = {
        "name": "quick_check",
        "links": [
            {"name": "link1", "length_m": 0.2, "mass_kg": 1.0},
        ],
        "joints": [
            {"name": "joint1", "joint_type": "revolute", "parent_link": "base_link",
             "child_link": "link1", "lower_limit_rad": -1.57, "upper_limit_rad": 1.57,
             "max_torque_nm": 10.0},
        ],
        "payload_mass_kg": 0.0,
    }

    response = client.post("/analyze/arm?include_lift_test=false", json=body)
    assert response.status_code == 200
    data = response.json()
    assert "lift_test" not in data
    assert "torque_check" in data
