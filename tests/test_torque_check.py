import pytest
from src.urdf_generator.samples import two_link_arm
from src.analysis.torque_check import compute_static_torques


def test_two_link_arm_torque_check_matches_hand_calculation():
    config = two_link_arm()
    results = compute_static_torques(config)

    assert len(results) == 2

    shoulder, elbow = results[0], results[1]

    assert shoulder["joint_name"] == "shoulder"
    assert shoulder["required_torque_nm"] == pytest.approx(9.07425, abs=0.001)
    assert shoulder["margin_percent"] == pytest.approx(39.5, abs=0.1)
    assert shoulder["passes"] is True

    assert elbow["joint_name"] == "elbow"
    assert elbow["required_torque_nm"] == pytest.approx(2.4525, abs=0.001)
    assert elbow["margin_percent"] == pytest.approx(69.3, abs=0.1)
    assert elbow["passes"] is True


def test_undersized_motor_fails_check():
    config = two_link_arm()
    config.joints[0].max_torque_nm = 5.0

    results = compute_static_torques(config)
    shoulder = results[0]

    assert shoulder["passes"] is False
    assert shoulder["margin_percent"] < 0
