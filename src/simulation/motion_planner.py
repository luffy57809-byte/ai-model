"""
Motion planning: given a target point, find joint angles (via inverse_kinematics.py),
generate a smooth motion from rest to that target, and check whether the
real motors can actually execute that motion.
"""

from src.urdf_generator.schema import ArmConfig
from src.analysis.inverse_kinematics import solve_ik
from src.simulation.trajectory_executor import execute_trajectory


def plan_reach_motion(
    config: ArmConfig,
    target_position: tuple[float, float, float],
    tracking_tolerance_rad: float = 0.05,
    record_trajectory: bool = True,
    trajectory_fps: int = 30,
) -> dict:
    ik_result = solve_ik(config, target_position)
    target_angles = ik_result["joint_angles_rad"]
    start_angles = [0.0] * len(config.joints)

    required_durations = []
    for joint, start, target in zip(config.joints, start_angles, target_angles):
        displacement = abs(target - start)
        if displacement < 1e-6:
            continue
        required_durations.append(1.5 * displacement / joint.max_velocity_rad_s)
    duration_s = max(max(required_durations, default=0.1) * 1.1, 0.1)

    sim_result = execute_trajectory(
        config, target_angles, duration_s,
        start_angles=start_angles,
        record_trajectory=record_trajectory,
        trajectory_fps=trajectory_fps,
    )

    overall_success = ik_result["reachable"]
    for jr in sim_result["joint_results"]:
        jr["motion_successful"] = jr["tracking_error_rad"] <= tracking_tolerance_rad
        overall_success = overall_success and jr["motion_successful"]

    return {
        "ik_result": ik_result,
        "duration_s": sim_result["duration_s"],
        "joint_results": sim_result["joint_results"],
        "overall_success": overall_success,
        **({"trajectory": sim_result["trajectory"]} if record_trajectory else {}),
    }
