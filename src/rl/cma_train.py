"""
CMA-ES policy search: searches directly over (target joint angles, duration)
to reach a target point - no IK involved.

This version is multi-objective: it can weight torque usage alongside
accuracy, which IK cannot do (IK only solves for a single static pose,
not a full motion's torque profile).
"""

import cma
import numpy as np

from src.simulation.trajectory_executor import execute_trajectory


def make_reward_fn(config, target_position, over_limit_penalty_weight=0.01, torque_usage_weight=0.0):
    """
    over_limit_penalty_weight: heavily discourages exceeding a joint's
    rated torque (safety - should always stay nonzero).
    torque_usage_weight: genuinely multi-objective term - prefers lower
    average torque usage across the whole motion, even within the safe
    range. Zero reproduces accuracy-only behavior.
    """
    n_joints = len(config.joints)

    def reward_fn(params: np.ndarray) -> float:
        target_angles = list(params[:n_joints])
        duration_s = float(np.clip(params[n_joints], 0.5, 3.0))

        result = execute_trajectory(
            config, target_angles, duration_s, record_trajectory=False
        )

        ee = np.array(result["final_end_effector_position"])
        target = np.array(target_position)
        distance = float(np.linalg.norm(ee - target))

        torque_ratios = [
            jr["max_applied_torque_nm"] / jr["rated_torque_nm"]
            for jr in result["joint_results"]
        ]
        max_torque_ratio = max(torque_ratios)
        over_limit_penalty = max(0.0, max_torque_ratio - 1.0) * over_limit_penalty_weight

        mean_torque_ratio = sum(torque_ratios) / len(torque_ratios)
        torque_usage_penalty = mean_torque_ratio * torque_usage_weight

        # Distance dominates: even a big torque saving shouldn't be worth
        # a real accuracy loss.
        accuracy_term = distance if distance < 0.02 else distance * 5

        return accuracy_term + over_limit_penalty + torque_usage_penalty

    return reward_fn


def random_baseline(config, target_position, n_trials=20, seed=0, torque_usage_weight=0.0):
    rng = np.random.default_rng(seed)
    n_joints = len(config.joints)
    reward_fn = make_reward_fn(config, target_position, torque_usage_weight=torque_usage_weight)

    best = float("inf")
    for _ in range(n_trials):
        angles = [
            rng.uniform(
                j.lower_limit_rad if j.lower_limit_rad is not None else -3.14159,
                j.upper_limit_rad if j.upper_limit_rad is not None else 3.14159,
            )
            for j in config.joints
        ]
        duration = rng.uniform(0.5, 3.0)
        score = reward_fn(np.array(angles + [duration]))
        best = min(best, score)
    return best


def train(config, target_position, generations=60, popsize=20, seed=0, torque_usage_weight=0.0):
    n_joints = len(config.joints)
    x0 = [0.0] * n_joints + [1.0]

    angle_stds = [
        max(1e-3, ((j.upper_limit_rad or 3.14159) - (j.lower_limit_rad or -3.14159)) / 4)
        for j in config.joints
    ]
    duration_std = 0.7
    stds = angle_stds + [duration_std]
    sigma0 = 1.0

    reward_fn = make_reward_fn(config, target_position, torque_usage_weight=torque_usage_weight)
    es = cma.CMAEvolutionStrategy(
        x0, sigma0, {"popsize": popsize, "seed": seed, "CMA_stds": stds}
    )

    history = []
    for gen in range(generations):
        solutions = es.ask()
        scores = [reward_fn(np.array(s)) for s in solutions]
        es.tell(solutions, scores)
        best_this_gen = min(scores)
        history.append(best_this_gen)
        print(f"gen {gen:3d}  best score: {best_this_gen:.4f}")

    return {
        "best_params": es.result.xbest,
        "best_score": es.result.fbest,
        "history": history,
    }


def report_solution(config, target_position, params, label=""):
    """Re-runs the given params and prints a human-readable breakdown:
    actual distance and per-joint torque usage, not just the combined
    scalar score."""
    n_joints = len(config.joints)
    target_angles = list(params[:n_joints])
    # Must match the exact clipping used in reward_fn during search, or
    # this prints a different motion than the one CMA-ES actually scored.
    duration_s = float(np.clip(params[n_joints], 0.5, 3.0))
    result = execute_trajectory(config, target_angles, duration_s, record_trajectory=False)

    ee = np.array(result["final_end_effector_position"])
    distance = float(np.linalg.norm(ee - np.array(target_position)))

    print(f"\n--- {label} ---")
    print(f"duration_s: {duration_s:.3f}")
    print(f"distance to target: {distance:.6f} m")
    for jr in result["joint_results"]:
        ratio = jr["max_applied_torque_nm"] / jr["rated_torque_nm"]
        print(
            f"  {jr['joint_name']}: angle={jr['target_angle_rad']:.3f} rad, "
            f"torque={jr['max_applied_torque_nm']:.3f}/{jr['rated_torque_nm']:.1f} Nm "
            f"({ratio*100:.1f}%)"
        )


if __name__ == "__main__":
    from src.urdf_generator.schema import ArmConfig, Link, Joint, JointType

    config = ArmConfig(
        name="reach_test",
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
    target = (0.4, 0.0, 0.85)

    print("=== Run 1: accuracy only (torque_usage_weight=0) ===")
    accuracy_result = train(config, target, torque_usage_weight=0.0)
    report_solution(config, target, accuracy_result["best_params"], label="Accuracy-only solution")

    print("\n=== Run 2: accuracy + torque minimization (torque_usage_weight=0.3) ===")
    torque_aware_result = train(config, target, torque_usage_weight=0.3)
    report_solution(config, target, torque_aware_result["best_params"], label="Torque-aware solution")
