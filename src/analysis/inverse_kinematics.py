"""
Inverse kinematics: given a target 3D point, find the joint angles that
place the arm's end effector there.

USES PYBULLET'S REAL, TESTED SOLVER (calculateInverseKinematics) rather
than a hand-rolled numerical method - same principle as using trimesh for
mesh math rather than deriving mesh integration ourselves.

VERIFIED, NOT TRUSTED BLINDLY: every solve_ik() call re-simulates the
returned joint angles and independently measures where the end effector
actually ends up. This caught a real finding during development: an arm
whose shoulder axis is parallel to its own link (a roll axis, not a lift
axis) is kinematically DEGENERATE for reaching - IK correctly returns a
large residual error for such an arm, because it genuinely can't reach
most points, not because the solver is broken.

END-EFFECTOR REFERENCE POINT: requires payload_mass_kg > 0, since
generator.py places a real payload link exactly at the chain's tip - the
most reliable point to target. Without a payload, the last actuated
link's reported position is its CENTER OF MASS, not its tip (an easy
mistake, caught during development). Supporting tip-targeting without a
payload is a reasonable future extension, not attempted here.
"""

import math
import tempfile
import os
import pybullet as p

from src.urdf_generator.schema import ArmConfig
from src.urdf_generator.generator import generate_urdf


class IKError(ValueError):
    """Raised when IK can't be attempted at all (e.g. no payload link to target)."""


def solve_ik(
    config: ArmConfig,
    target_position: tuple[float, float, float],
    tolerance_m: float = 0.01,
    max_iterations: int = 200,
) -> dict:
    if config.payload_mass_kg <= 0:
        raise IKError(
            "Inverse kinematics currently requires payload_mass_kg > 0, "
            "since the payload link sits exactly at the chain's tip - the "
            "most reliable point to target. Add a small payload_mass_kg "
            "(e.g. 0.01) even if you don't have a real payload, to give "
            "IK a tip to reach for."
        )

    urdf_string = generate_urdf(config)

    physics_client = p.connect(p.DIRECT)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(urdf_string)
            urdf_path = f.name

        try:
            robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.5], useFixedBase=True)
        finally:
            os.unlink(urdf_path)

        num_joints = p.getNumJoints(robot_id)

        payload_index = None
        actuated_indices = []
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            link_name = info[12].decode("utf-8")
            joint_type = info[2]
            if link_name == "payload":
                payload_index = i
            if joint_type != p.JOINT_FIXED:
                actuated_indices.append(i)

        if payload_index is None:
            raise IKError("Could not find the payload link in the generated URDF.")

        ik_solution = p.calculateInverseKinematics(
            robot_id, payload_index, list(target_position),
            maxNumIterations=max_iterations, residualThreshold=1e-6,
        )

        for idx, angle in zip(actuated_indices, ik_solution):
            p.resetJointState(robot_id, idx, angle)
        p.stepSimulation()

        achieved_state = p.getLinkState(robot_id, payload_index)
        achieved_position = list(achieved_state[0])
        error = math.dist(achieved_position, target_position)

        return {
            "joint_angles_rad": [round(a, 5) for a in ik_solution],
            "achieved_position_m": [round(v, 5) for v in achieved_position],
            "target_position_m": list(target_position),
            "position_error_m": round(error, 5),
            "reachable": error <= tolerance_m,
        }
    finally:
        p.disconnect(physics_client)
