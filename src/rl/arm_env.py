"""
A minimal step-by-step RL environment for the 2-link reach_test arm,
wrapping PyBullet directly - unlike trajectory_executor.py (which
interpolates a predetermined smoothstep trajectory to fixed target
angles), this exposes true sequential control: the policy chooses a
torque command at EVERY timestep based on the current state, the way
real RL problems are actually posed.

Observation: [shoulder_angle, shoulder_velocity, elbow_angle,
              elbow_velocity, target_x, target_y, target_z] (7-dim)
Action: [shoulder_torque_fraction, elbow_torque_fraction], each in
        [-1, 1], scaled by each joint's max_torque_nm
Reward: shaped - negative distance to target each step (dense signal,
        not just a final sparse reward), so the policy gets useful
        gradient information throughout the episode, not just at the end.
"""

import math

import numpy as np
import pybullet as p

from src.urdf_generator.schema import ArmConfig, Link, Joint, JointType
from src.urdf_generator.generator import generate_urdf


def default_config() -> ArmConfig:
    """Same reach_test config used throughout this project."""
    return ArmConfig(
        name="reach_test",
        links=[
            Link(name="upper_arm", length_m=0.3, mass_kg=1.5),
            Link(name="forearm", length_m=0.25, mass_kg=1.0),
        ],
        joints=[
            Joint(name="shoulder", joint_type=JointType.REVOLUTE, parent_link="base_link",
                  child_link="upper_arm", axis=(0, 1, 0), lower_limit_rad=-1.57,
                  upper_limit_rad=1.57, max_torque_nm=15.0, max_velocity_rad_s=5.0),
            Joint(name="elbow", joint_type=JointType.REVOLUTE, parent_link="upper_arm",
                  child_link="forearm", axis=(0, 1, 0), lower_limit_rad=-2.5,
                  upper_limit_rad=2.5, max_torque_nm=8.0, max_velocity_rad_s=5.0),
        ],
        payload_mass_kg=0.5,
    )


class ArmReachEnv:
    def __init__(self, config: ArmConfig = None, max_steps: int = 240, target_position=(0.4, 0.0, 0.85)):
        self.config = config or default_config()
        self.max_steps = max_steps
        self.target_position = np.array(target_position, dtype=np.float64)

        self._physics_client = None
        self._robot_id = None
        self._joint_indices = None
        self._max_torques = None
        self._step_count = 0

    def _build_robot(self):
        import tempfile, os
        urdf_string = generate_urdf(self.config)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(urdf_string)
            urdf_path = f.name
        try:
            robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.5], useFixedBase=True)
        finally:
            os.unlink(urdf_path)

        num_joints = p.getNumJoints(robot_id)
        actuated = []
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_type != p.JOINT_FIXED:
                actuated.append((i, name))

        joint_indices = [next(idx for idx, name in actuated if name == j.name) for j in self.config.joints]
        max_torques = [j.max_torque_nm for j in self.config.joints]

        # Find payload link for reward computation (same convention as
        # inverse_kinematics.py / trajectory_executor.py).
        payload_index = None
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            if info[12].decode("utf-8") == "payload":
                payload_index = i
                break

        return robot_id, joint_indices, max_torques, payload_index

    def reset(self) -> np.ndarray:
        if self._physics_client is not None:
            p.disconnect(self._physics_client)

        self._physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        self._robot_id, self._joint_indices, self._max_torques, self._payload_index = self._build_robot()
        self._step_count = 0

        # Disable default motor control so torque commands actually apply
        # freely (PyBullet's velocity motors resist torque control otherwise).
        for idx in self._joint_indices:
            p.setJointMotorControl2(self._robot_id, idx, p.VELOCITY_CONTROL, force=0)

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        obs = []
        for idx in self._joint_indices:
            state = p.getJointState(self._robot_id, idx)
            obs.append(state[0])  # angle
            obs.append(state[1])  # velocity
        obs.extend(self.target_position.tolist())
        return np.array(obs, dtype=np.float64)

    def _get_ee_position(self) -> np.ndarray:
        state = p.getLinkState(self._robot_id, self._payload_index)
        return np.array(state[0])

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        for idx, max_torque, a in zip(self._joint_indices, self._max_torques, action):
            p.setJointMotorControl2(
                self._robot_id, idx, p.TORQUE_CONTROL, force=float(a * max_torque)
            )
        p.stepSimulation()
        self._step_count += 1

        obs = self._get_observation()
        ee_position = self._get_ee_position()
        distance = float(np.linalg.norm(ee_position - self.target_position))

        # Dense shaped reward: negative distance each step, small control
        # penalty to discourage needless large torques.
        control_penalty = 0.001 * float(np.sum(action ** 2))
        reward = -distance - control_penalty

        done = self._step_count >= self.max_steps
        info = {"distance": distance, "ee_position": ee_position.tolist()}
        return obs, reward, done, info

    def close(self):
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
            self._physics_client = None
