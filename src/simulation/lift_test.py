"""
Dynamic "hold horizontal, fully extended, under payload" test.

CRITICAL MESH-LINK FIX: PyBullet's URDF loader silently IGNORES the
<inertia> values we write for mesh-collision-geometry links, recomputing
its own instead (confirmed: two URDFs differing only in specified inertia
produced identical simulated dynamics). Fix: after loadURDF, force-override
via p.changeDynamics(mass=..., localInertiaDiagonal=...) using our real,
trimesh-verified values.

KNOWN LIMITATION: changeDynamics() has no orientation parameter in this
PyBullet version, so this only correctly handles near-diagonal inertia
tensors (the common case). A mesh with a genuinely tilted principal-axis
mass distribution raises MeshDynamicsError rather than silently producing
wrong dynamics. The static torque_check is unaffected either way.
"""

import math
import tempfile
import os
import pybullet as p

from src.urdf_generator.schema import ArmConfig
from src.urdf_generator.generator import generate_urdf


class MeshDynamicsError(ValueError):
    """Raised when a mesh link's inertia can't be safely applied in PyBullet."""


def _has_significant_off_diagonal_terms(link, relative_threshold: float = 0.01) -> bool:
    diag_scale = max(abs(link.inertia_ixx), abs(link.inertia_iyy), abs(link.inertia_izz), 1e-12)
    off_diag_max = max(abs(link.inertia_ixy), abs(link.inertia_ixz), abs(link.inertia_iyz))
    return (off_diag_max / diag_scale) > relative_threshold


def _apply_real_mesh_dynamics(robot_id: int, link_index: int, link) -> None:
    if _has_significant_off_diagonal_terms(link):
        raise MeshDynamicsError(
            f"Link '{link.name}' has a significantly off-diagonal inertia tensor "
            f"(a tilted principal-axis mass distribution). This PyBullet version's "
            f"changeDynamics() can't correctly represent that via localInertiaDiagonal "
            f"alone, so the dynamic lift test would silently misassign inertia to the "
            f"wrong axes. The static torque check is unaffected. Consider reorienting "
            f"the mesh, or skip the dynamic lift test for this design."
        )
    p.changeDynamics(
        robot_id, link_index,
        mass=link.mass_kg,
        localInertiaDiagonal=(link.inertia_ixx, link.inertia_iyy, link.inertia_izz),
    )


def run_lift_test(
    config: ArmConfig,
    sim_seconds: float = 3.0,
    sag_tolerance_deg: float = 5.0,
    record_trajectory: bool = False,
    trajectory_fps: int = 30,
) -> dict:
    urdf_string = generate_urdf(config)

    physics_client = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(urdf_string)
            urdf_path = f.name

        base_orientation = p.getQuaternionFromEuler([0, math.pi / 2, 0])

        try:
            robot_id = p.loadURDF(
                urdf_path, basePosition=[0, 0, 0.5],
                baseOrientation=base_orientation, useFixedBase=True,
            )
        finally:
            os.unlink(urdf_path)

        num_joints = p.getNumJoints(robot_id)
        actuated_joint_indices = []
        for i in range(num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_name = joint_info[1].decode("utf-8")
            joint_type = joint_info[2]
            if joint_type != p.JOINT_FIXED:
                actuated_joint_indices.append((i, joint_name))

        config_joint_by_name = {j.name: j for j in config.joints}
        config_link_by_name = {l.name: l for l in config.links}
        max_torque_tracker = {idx: 0.0 for idx, _ in actuated_joint_indices}

        for link_index in range(num_joints):
            link_info = p.getJointInfo(robot_id, link_index)
            link_name = link_info[12].decode("utf-8")
            link_config = config_link_by_name.get(link_name)
            if link_config is not None and link_config.is_mesh_based():
                _apply_real_mesh_dynamics(robot_id, link_index, link_config)

        for idx, name in actuated_joint_indices:
            cfg_joint = config_joint_by_name[name]
            p.setJointMotorControl2(
                bodyUniqueId=robot_id, jointIndex=idx, controlMode=p.POSITION_CONTROL,
                targetPosition=0.0, force=cfg_joint.max_torque_nm,
                maxVelocity=cfg_joint.max_velocity_rad_s,
            )

        ordered_indices = [
            next(idx for idx, name in actuated_joint_indices if name == j.name)
            for j in config.joints
        ]

        steps = int(sim_seconds * 240)
        record_every = max(1, int(240 / trajectory_fps)) if record_trajectory else None
        trajectory_frames = [] if record_trajectory else None

        for step in range(steps):
            p.stepSimulation()
            for idx, _ in actuated_joint_indices:
                applied_torque = abs(p.getJointState(robot_id, idx)[3])
                if applied_torque > max_torque_tracker[idx]:
                    max_torque_tracker[idx] = applied_torque
            if record_trajectory and step % record_every == 0:
                frame = [p.getJointState(robot_id, idx)[0] for idx in ordered_indices]
                trajectory_frames.append(frame)

        joint_results = []
        overall_passes = True
        for idx, name in actuated_joint_indices:
            cfg_joint = config_joint_by_name[name]
            final_angle_rad = p.getJointState(robot_id, idx)[0]
            final_angle_deg = math.degrees(final_angle_rad)
            sag_deg = abs(final_angle_deg)
            passes = sag_deg <= sag_tolerance_deg
            overall_passes = overall_passes and passes
            joint_results.append({
                "joint_name": name, "target_angle_deg": 0.0,
                "final_angle_deg": round(final_angle_deg, 2), "sag_deg": round(sag_deg, 2),
                "max_applied_torque_nm": round(max_torque_tracker[idx], 3),
                "rated_max_torque_nm": cfg_joint.max_torque_nm, "passes": passes,
            })

        result = {"joint_results": joint_results, "overall_passes": overall_passes}
        if record_trajectory:
            result["trajectory"] = {
                "fps": trajectory_fps, "joint_order": [j.name for j in config.joints],
                "frames": trajectory_frames,
            }
        return result
    finally:
        p.disconnect(physics_client)
