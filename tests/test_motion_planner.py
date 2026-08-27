import pytest
from src.urdf_generator.schema import ArmConfig, Link, Joint, JointType
from src.simulation.motion_planner import plan_reach_motion
from src.simulation.trajectory_executor import _smoothstep


def _properly_articulated_arm(shoulder_torque=15.0, elbow_torque=8.0):
    return ArmConfig(
        name="reach_test",
        links=[
            Link(name="upper_arm", length_m=0.3, mass_kg=1.5),
            Link(name="forearm", length_m=0.25, mass_kg=1.0),
        ],
        joints=[
            Joint(name="shoulder", joint_type=JointType.REVOLUTE, parent_link="base_link",
                  child_link="upper_arm", axis=(0, 1, 0), lower_limit_rad=-1.57,
                  upper_limit_rad=1.57, max_torque_nm=shoulder_torque),
            Joint(name="elbow", joint_type=JointType.REVOLUTE, parent_link="upper_arm",
                  child_link="forearm", axis=(0, 1, 0), lower_limit_rad=-2.5,
                  upper_limit_rad=2.5, max_torque_nm=elbow_torque),
        ],
        payload_mass_kg=0.5,
    )


def test_smoothstep_boundary_and_midpoint_values():
    assert _smoothstep(0.0) == 0.0
    assert _smoothstep(1.0) == 1.0
    assert _smoothstep(0.5) == pytest.approx(0.5, abs=1e-9)
    assert _smoothstep(1.5) == 1.0
    assert _smoothstep(-0.5) == 0.0


def test_reachable_target_with_adequate_motors_succeeds():
    config = _properly_articulated_arm()
    result = plan_reach_motion(config, target_position=(0.4, 0.0, 0.85))
    assert result["ik_result"]["reachable"] is True
    assert result["overall_success"] is True
    for joint_result in result["joint_results"]:
        assert joint_result["motion_successful"] is True
        assert joint_result["tracking_error_rad"] < 0.05


def test_undersized_motor_fails_to_track_and_is_correctly_flagged():
    config = _properly_articulated_arm(shoulder_torque=1.0)
    result = plan_reach_motion(config, target_position=(0.4, 0.0, 0.85), record_trajectory=False)
    shoulder_result = result["joint_results"][0]
    assert shoulder_result["motion_successful"] is False
    assert shoulder_result["tracking_error_rad"] > 0.1
    assert result["overall_success"] is False


def test_trajectory_starts_near_zero_and_ends_near_target():
    config = _properly_articulated_arm()
    result = plan_reach_motion(config, target_position=(0.4, 0.0, 0.85), record_trajectory=True)
    frames = result["trajectory"]["frames"]
    target_angles = result["ik_result"]["joint_angles_rad"]
    assert abs(frames[0][0]) < 0.01
    assert abs(frames[0][1]) < 0.01
    assert abs(frames[-1][0] - target_angles[0]) < 0.1
    assert abs(frames[-1][1] - target_angles[1]) < 0.1


def test_unreachable_target_still_returns_a_result_but_flags_ik_unreachable():
    config = _properly_articulated_arm()
    result = plan_reach_motion(config, target_position=(5.0, 0.0, 0.5), record_trajectory=False)
    assert result["ik_result"]["reachable"] is False
    assert result["overall_success"] is False


def test_duration_scales_with_slower_max_velocity():
    fast_config = _properly_articulated_arm()
    result_fast = plan_reach_motion(fast_config, target_position=(0.4, 0.0, 0.85), record_trajectory=False)

    slow_config = _properly_articulated_arm()
    for joint in slow_config.joints:
        joint.max_velocity_rad_s = joint.max_velocity_rad_s / 10.0
    result_slow = plan_reach_motion(slow_config, target_position=(0.4, 0.0, 0.85), record_trajectory=False)

    assert result_slow["duration_s"] > result_fast["duration_s"] * 5
