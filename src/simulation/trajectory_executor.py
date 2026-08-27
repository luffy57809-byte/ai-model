"""
Shared simulation core: given a config, a set of target joint angles, and a
duration, executes a smoothstep trajectory in PyBullet and reports what
actually happened (tracking error, torque, end-effector position).

Used by both motion_planner.py (IK-driven reaching) and cma_train.py
(policy search - target angles come from the optimizer, not IK).
"""

import tempfile
import os
import pybullet as p

from src.urdf_generator.schema import ArmConfig
from src.urdf_generator.generator import generate_urdf
from src.simulation.lift_test import _apply_real_mesh_dynamics


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 3 * t**2 - 2 * t**3


def execute_trajectory(
    config: ArmConfig,
    target_angles: list[float],
    duration_s: float,
    start_angles: list[float] | None = None,
    record_trajectory: bool = True,
    trajectory_fps: int = 30,
) -> dict:
    if start_angles is None:
        start_angles = [0.0] * len(config.joints)

    urdf_string = generate_urdf(config)
    physics_client = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(urdf_string)
            urdf_path = f.name
        try:
            robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.5], useFixedBase=True)
        finally:
            os.unlink(urdf_path)

        num_joints = p.getNumJoints(robot_id)
        actuated_joint_indices = []
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_type != p.JOINT_FIXED:
                actuated_joint_indices.append((i, name))

        config_link_by_name = {l.name: l for l in config.links}
        for link_index in range(num_joints):
            link_info = p.getJointInfo(robot_id, link_index)
            link_name = link_info[12].decode("utf-8")
            link_config = config_link_by_name.get(link_name)
            if link_config is not None and link_config.is_mesh_based():
                _apply_real_mesh_dynamics(robot_id, link_index, link_config)

        ordered_indices = [
            next(idx for idx, name in actuated_joint_indices if name == j.name)
            for j in config.joints
        ]

        # The true end-effector reference point is the "payload" link that
        # generator.py places at the chain's tip (same convention as
        # inverse_kinematics.py) - NOT the last actuated joint's own origin,
        # which is the proximal end of the last link, not its tip.
        payload_index = None
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            link_name = info[12].decode("utf-8")
            if link_name == "payload":
                payload_index = i
                break
        if payload_index is None:
            raise ValueError(
                "Could not find the payload link in the generated URDF - "
                "execute_trajectory requires payload_mass_kg > 0, same as solve_ik."
            )

        max_torque_tracker = {idx: 0.0 for idx, _ in actuated_joint_indices}

        steps = int(duration_s * 240)
        record_every = max(1, int(240 / trajectory_fps)) if record_trajectory else None
        trajectory_frames = [] if record_trajectory else None

        for step in range(steps):
            t = step / steps if steps > 0 else 1.0
            profile = _smoothstep(t)

            for idx, start, target, joint in zip(ordered_indices, start_angles, target_angles, config.joints):
                commanded_angle = start + (target - start) * profile
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id, jointIndex=idx, controlMode=p.POSITION_CONTROL,
                    targetPosition=commanded_angle, force=joint.max_torque_nm,
                    maxVelocity=joint.max_velocity_rad_s,
                )

            p.stepSimulation()

            for idx, _ in actuated_joint_indices:
                applied_torque = abs(p.getJointState(robot_id, idx)[3])
                if applied_torque > max_torque_tracker[idx]:
                    max_torque_tracker[idx] = applied_torque

            if record_trajectory and step % record_every == 0:
                frame = [p.getJointState(robot_id, idx)[0] for idx in ordered_indices]
                trajectory_frames.append(frame)

        joint_results = []
        for idx, target_angle, joint in zip(ordered_indices, target_angles, config.joints):
            achieved_angle = p.getJointState(robot_id, idx)[0]
            joint_results.append({
                "joint_name": joint.name,
                "target_angle_rad": round(target_angle, 5),
                "achieved_angle_rad": round(achieved_angle, 5),
                "tracking_error_rad": round(abs(achieved_angle - target_angle), 5),
                "max_applied_torque_nm": round(max_torque_tracker[idx], 3),
                "rated_torque_nm": joint.max_torque_nm,
            })

        final_ee_position = p.getLinkState(robot_id, payload_index)[0]

        result = {
            "duration_s": round(duration_s, 3),
            "joint_results": joint_results,
            "final_end_effector_position": final_ee_position,
        }

        if record_trajectory:
            result["trajectory"] = {
                "fps": trajectory_fps,
                "joint_order": [j.name for j in config.joints],
                "frames": trajectory_frames,
            }

        return result
    finally:
        p.disconnect(physics_client)
