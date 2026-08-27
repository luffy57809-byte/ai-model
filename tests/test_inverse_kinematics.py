import pytest
from src.urdf_generator.schema import ArmConfig, Link, Joint, JointType
from src.analysis.inverse_kinematics import solve_ik, IKError


def _properly_articulated_arm():
    return ArmConfig(
        name="proper_reach_test",
        links=[
            Link(name="upper_arm", length_m=0.3, mass_kg=1.5),
            Link(name="forearm", length_m=0.25, mass_kg=1.0),
        ],
        joints=[
            Joint(name="shoulder", joint_type=JointType.REVOLUTE, parent_link="base_link",
                  child_link="upper_arm", axis=(0, 1, 0), lower_limit_rad=-1.57,
                  upper_limit_rad=1.57, max_torque_nm=15.0),
            Joint(name="elbow", joint_type=JointType.REVOLUTE, parent_link="upper_arm",
                  child_link="forearm", axis=(0, 1, 0), lower_limit_rad=-2.5,
                  upper_limit_rad=2.5, max_torque_nm=8.0),
        ],
        payload_mass_kg=0.5,
    )


def test_requires_payload_to_attempt_ik():
    config = _properly_articulated_arm()
    config.payload_mass_kg = 0.0
    with pytest.raises(IKError, match="requires payload_mass_kg"):
        solve_ik(config, target_position=(0.4, 0.0, 0.85))


def test_reachable_target_converges_accurately():
    config = _properly_articulated_arm()
    target = (0.4, 0.0, 0.85)
    result = solve_ik(config, target_position=target, tolerance_m=0.01)
    assert result["reachable"] is True
    assert result["position_error_m"] < 0.01
    assert len(result["joint_angles_rad"]) == 2


def test_rest_position_matches_hand_calculation():
    config = _properly_articulated_arm()
    result = solve_ik(config, target_position=(0.0, 0.0, 1.05), tolerance_m=0.005)
    assert result["reachable"] is True
    for angle in result["joint_angles_rad"]:
        assert abs(angle) < 0.05


def test_unreachable_target_far_outside_arm_reach_is_flagged():
    config = _properly_articulated_arm()
    result = solve_ik(config, target_position=(5.0, 0.0, 0.5), tolerance_m=0.01)
    assert result["reachable"] is False
    assert result["position_error_m"] > 0.1


def test_degenerate_roll_axis_arm_has_large_residual_error():
    degenerate_config = ArmConfig(
        name="degenerate_roll_test",
        links=[
            Link(name="upper_arm", length_m=0.3, mass_kg=1.5),
            Link(name="forearm", length_m=0.25, mass_kg=1.0),
        ],
        joints=[
            Joint(name="shoulder", joint_type=JointType.REVOLUTE, parent_link="base_link",
                  child_link="upper_arm", axis=(0, 0, 1),
                  lower_limit_rad=-3.14, upper_limit_rad=3.14, max_torque_nm=15.0),
            Joint(name="elbow", joint_type=JointType.REVOLUTE, parent_link="upper_arm",
                  child_link="forearm", axis=(0, 1, 0), lower_limit_rad=-2.5,
                  upper_limit_rad=2.5, max_torque_nm=8.0),
        ],
        payload_mass_kg=0.5,
    )
    result = solve_ik(degenerate_config, target_position=(0.4, 0.0, 0.85), tolerance_m=0.01)
    assert result["reachable"] is False
    assert result["position_error_m"] > 0.05
