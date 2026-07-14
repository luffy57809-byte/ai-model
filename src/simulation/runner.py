"""
Minimal PyBullet simulation runner.

Phase 1 goal: prove the pipeline works end-to-end (config -> URDF -> physics
engine -> real numbers out), not to implement every test scenario yet.
Runs headless (p.DIRECT) so it works in a Codespace with no display.
"""

import tempfile
import os
import pybullet as p


def run_smoke_test(urdf_string: str, sim_seconds: float = 2.0) -> dict:
    """
    Loads the given URDF into a headless physics world, lets it settle
    under gravity for `sim_seconds`, and returns basic real outputs.

    Returns a dict with:
      - final_base_position: [x, y, z]
      - final_joint_positions: {joint_index: angle_rad}
    """
    physics_client = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(urdf_string)
            urdf_path = f.name

        try:
            robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
        finally:
            os.unlink(urdf_path)

        num_joints = p.getNumJoints(robot_id)
        steps = int(sim_seconds * 240)
        for _ in range(steps):
            p.stepSimulation()

        base_pos, base_orient = p.getBasePositionAndOrientation(robot_id)
        joint_positions = {}
        for i in range(num_joints):
            joint_state = p.getJointState(robot_id, i)
            joint_positions[i] = joint_state[0]

        return {
            "num_joints": num_joints,
            "final_base_position": list(base_pos),
            "final_joint_positions": joint_positions,
        }
    finally:
        p.disconnect(physics_client)
