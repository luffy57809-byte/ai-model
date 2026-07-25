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


def test_trajectory_not_recorded_by_default():
    config = two_link_arm()
    result = run_lift_test(config, sim_seconds=0.5)
    assert "trajectory" not in result


def test_trajectory_recording_produces_expected_frame_count_and_order():
    config = two_link_arm()
    result = run_lift_test(config, sim_seconds=2.0, record_trajectory=True, trajectory_fps=30)

    assert "trajectory" in result
    traj = result["trajectory"]
    assert traj["joint_order"] == ["shoulder", "elbow"]
    assert len(traj["frames"]) == 60
    assert all(len(frame) == 2 for frame in traj["frames"])


def test_trajectory_captures_real_motion_for_an_undersized_joint():
    config = two_link_arm()
    config.joints[1].max_torque_nm = 0.5

    result = run_lift_test(config, sim_seconds=2.0, record_trajectory=True, trajectory_fps=30)
    elbow_angles = [frame[1] for frame in result["trajectory"]["frames"]]

    assert abs(elbow_angles[0]) < 0.1
    assert max(abs(a) for a in elbow_angles) > 0.5
