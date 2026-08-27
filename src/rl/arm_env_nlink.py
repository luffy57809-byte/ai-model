"""
N-link generalization of arm_env.py's ArmReachEnv - supports arms with
2-5 links via a FIXED-SIZE padded observation/action space, so the same
policy architecture works regardless of the actual arm's link count.

Observation (18-dim, fixed): [angle_1, vel_1, ..., angle_5, vel_5,
mask_1, ..., mask_5, target_x, target_y, target_z]. mask_i is 1.0 if
joint i is real, 0.0 if it's padding - lets the policy learn to
distinguish real state from padding rather than being fed misleading
zeros with no way to tell them apart from a genuine zero angle.

Action (5-dim, fixed): policy always outputs 5 torque-fraction values;
the environment only applies the first N to the arm's N real joints and
ignores the rest.

Each episode can use a DIFFERENT randomly-sampled link count - this is
what actually tests generalization, as opposed to training on one fixed
arm padded to 5 slots (which would prove nothing about generalizing
across different structures).
"""

import numpy as np
import pybullet as p

from src.urdf_generator.schema import ArmConfig, Link, Joint, JointType
from src.urdf_generator.generator import generate_urdf

MAX_LINKS = 5
OBS_DIM = MAX_LINKS * 2 + MAX_LINKS + 3  # angles+velocities, mask, target xyz
ACTION_DIM = MAX_LINKS


def random_config(rng: np.random.Generator, n_links: int = None) -> ArmConfig:
    if n_links is None:
        n_links = int(rng.integers(2, MAX_LINKS + 1))

    links, joints = [], []
    parent = "base_link"
    for i in range(n_links):
        length_m = float(rng.uniform(0.15, 0.35))
        mass_kg = float(rng.uniform(0.3, 2.0))
        torque_nm = float(rng.uniform(5.0, 20.0))

        links.append(Link(name=f"link{i}", length_m=length_m, mass_kg=mass_kg, radius_m=0.03))
        joints.append(Joint(
            name=f"joint{i}", joint_type=JointType.REVOLUTE, parent_link=parent,
            child_link=f"link{i}", axis=(0, 1, 0), lower_limit_rad=-1.57, upper_limit_rad=1.57,
            max_torque_nm=torque_nm, max_velocity_rad_s=5.0,
        ))
        parent = f"link{i}"

    return ArmConfig(name="nlink_rl_sample", links=links, joints=joints, payload_mass_kg=0.3)


class NLinkArmReachEnv:
    def __init__(self, max_steps: int = 150, target_position=(0.4, 0.0, 0.85), seed: int = 0):
        self.max_steps = max_steps
        self.target_position = np.array(target_position, dtype=np.float64)
        self._rng = np.random.default_rng(seed)

        self._physics_client = None
        self._robot_id = None
        self._joint_indices = None
        self._max_torques = None
        self._payload_index = None
        self._n_links = None
        self._step_count = 0

    def _build_robot(self, config: ArmConfig):
        import tempfile, os
        urdf_string = generate_urdf(config)
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

        joint_indices = [next(idx for idx, name in actuated if name == j.name) for j in config.joints]
        max_torques = [j.max_torque_nm for j in config.joints]

        payload_index = None
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            if info[12].decode("utf-8") == "payload":
                payload_index = i
                break

        return robot_id, joint_indices, max_torques, payload_index

    def reset(self, n_links: int = None) -> np.ndarray:
        if self._physics_client is not None:
            p.disconnect(self._physics_client)

        self._physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)

        config = random_config(self._rng, n_links=n_links)
        self._n_links = len(config.links)
        self._robot_id, self._joint_indices, self._max_torques, self._payload_index = self._build_robot(config)
        self._step_count = 0

        for idx in self._joint_indices:
            p.setJointMotorControl2(self._robot_id, idx, p.VELOCITY_CONTROL, force=0)

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        angles_vels = np.zeros(MAX_LINKS * 2)
        mask = np.zeros(MAX_LINKS)
        for i, idx in enumerate(self._joint_indices):
            state = p.getJointState(self._robot_id, idx)
            angles_vels[2 * i] = state[0]
            angles_vels[2 * i + 1] = state[1]
            mask[i] = 1.0
        return np.concatenate([angles_vels, mask, self.target_position])

    def _get_ee_position(self) -> np.ndarray:
        state = p.getLinkState(self._robot_id, self._payload_index)
        return np.array(state[0])

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        # Only apply the first n_links actions - the rest are padding
        # the environment ignores, matching the fixed-size action head.
        for idx, max_torque, a in zip(self._joint_indices, self._max_torques, action[: self._n_links]):
            p.setJointMotorControl2(self._robot_id, idx, p.TORQUE_CONTROL, force=float(a * max_torque))
        p.stepSimulation()
        self._step_count += 1

        obs = self._get_observation()
        ee_position = self._get_ee_position()
        distance = float(np.linalg.norm(ee_position - self.target_position))

        control_penalty = 0.001 * float(np.sum(action[: self._n_links] ** 2))
        reward = -distance - control_penalty

        done = self._step_count >= self.max_steps
        info = {"distance": distance, "n_links": self._n_links}
        return obs, reward, done, info

    def close(self):
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
            self._physics_client = None
