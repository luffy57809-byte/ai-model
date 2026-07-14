"""
Dynamic "hold horizontal, fully extended, under payload" test.

This is the dynamic counterpart to analysis/torque_check.py's static
calculation. Same worst-case pose (arm fully extended, horizontal,
payload at the tip) - but here PyBullet's actual motor control and
gravity decide what happens, instead of a hand-derived equation.

How the horizontal pose is achieved: the URDF chain extends along its
own local z-axis by construction (see generator.py). We mount the base
rotated 90 degrees about Y, so that local-z-forward becomes
horizontal in world space. With all joint targets held at 0 (i.e.
"stay straight"), this reproduces the same fully-extended-horizontal
pose the static check assumes - so the two are directly comparable.

Each joint is driven with PyBullet's built-in POSITION_CONTROL, capped
at that joint's real max_torque_nm as the force limit. If the motor is
strong enough, it holds near 0 rad. If it's undersized, gravity wins and
the joint sags - which is exactly the failure mode we want to catch.
"""

import math
import tempfile
import os
import pybullet as p

from src.urdf_generator.schema import ArmConfig
from src.urdf_generator.generator import generate_urdf


def run_lift_test(config: ArmConfig, sim_seconds: float = 3.0, sag_tolerance_deg: float = 5.0) -> dict:
    """
    Returns:
      {
        "joint_results": [
          {
            "joint_name": str,
            "target_angle_deg": 0.0,
            "final_angle_deg": float,
            "sag_deg": float,
            "max_applied_torque_nm": float,
            "rated_max_torque_nm": float,
            "passes": bool,
          },
          ...
        ],
        "overall_passes": bool,
      }
    """
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
                urdf_path,
                basePosition=[0, 0, 0.5],
                baseOrientation=base_orientation,
                useFixedBase=True,
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
        max_torque_tracker = {idx: 0.0 for idx, _ in actuated_joint_indices}

        for idx, name in actuated_joint_indices:
            cfg_joint = config_joint_by_name[name]
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
                force=cfg_joint.max_torque_nm,
                maxVelocity=cfg_joint.max_velocity_rad_s,
            )

        steps = int(sim_seconds * 240)
        for _ in range(steps):
            p.stepSimulation()
            for idx, _ in actuated_joint_indices:
                applied_torque = abs(p.getJointState(robot_id, idx)[3])
                if applied_torque > max_torque_tracker[idx]:
                    max_torque_tracker[idx] = applied_torque

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
                "joint_name": name,
                "target_angle_deg": 0.0,
                "final_angle_deg": round(final_angle_deg, 2),
                "sag_deg": round(sag_deg, 2),
                "max_applied_torque_nm": round(max_torque_tracker[idx], 3),
                "rated_max_torque_nm": cfg_joint.max_torque_nm,
                "passes": passes,
            })

        return {
            "joint_results": joint_results,
            "overall_passes": overall_passes,
        }
    finally:
        p.disconnect(physics_client)
