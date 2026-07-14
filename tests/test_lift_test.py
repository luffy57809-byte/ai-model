import pytest
from src.urdf_generator.samples import two_link_arm
from src.simulation.lift_test import run_lift_test


def test_elbow_dynamic_torque_matches_static_calculation():
    """
    The elbow's axis (0,1,0) is perpendicular to its link's own length axis,
    so it genuinely has to fight gravity to hold the arm extended. Expect
    close agreement with the static result of ~2.4525 Nm.
    """
    config = two_link_arm()
    result = run_lift_test(config)

    elbow = next(j for j in result["joint_results"] if j["joint_name"] == "elbow")
    assert elbow["max_applied_torque_nm"] == pytest.approx(2.4525, abs=0.05)
    assert elbow["passes"] is True


def test_shoulder_roll_axis_requires_near_zero_torque():
    """
    Documents a real finding: the shoulder's axis (0,0,1) is parallel to its
    own link's length, making it a roll/twist joint rather than a lifting
    joint. Rotating a symmetric link about its own axis doesn't fight
    gravity, so the dynamic sim correctly shows ~0 applied torque here -
    even though the static check (which doesn't model axis direction)
    estimated ~9 Nm required. Known limitation of the static check.
    """
    config = two_link_arm()
    result = run_lift_test(config)

    shoulder = next(j for j in result["joint_results"] if j["joint_name"] == "shoulder")
    assert shoulder["max_applied_torque_nm"] < 0.5
    assert shoulder["passes"] is True


def test_undersized_elbow_sags_and_fails():
    """If the elbow motor is too weak, it should genuinely sag under the payload."""
    config = two_link_arm()
    config.joints[1].max_torque_nm = 0.5

    result = run_lift_test(config)
    elbow = next(j for j in result["joint_results"] if j["joint_name"] == "elbow")

    assert elbow["sag_deg"] > 5.0
    assert elbow["passes"] is False
    assert result["overall_passes"] is False
