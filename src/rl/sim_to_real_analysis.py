"""
Sim-to-real gap analysis: since we have no real hardware to compare
against, this estimates robustness the honest way - systematically
perturbing simulation parameters a real robot would inevitably differ
on (mass, torque, sensor noise, control delay), and measuring how much
the TRAINED policy's performance degrades under each. Large degradation
under a given perturbation = that's a real thing worth being careful
about (or specifically robustifying against, e.g. via domain
randomization) before trusting the policy on physical hardware.

This does NOT claim to predict real-world performance exactly - it's a
sensitivity/robustness proxy, not a guarantee. That distinction matters
and is reported explicitly in the output.
"""

import numpy as np

from src.rl.arm_env import ArmReachEnv, default_config
from src.rl.policy import GaussianPolicy
from src.urdf_generator.schema import ArmConfig, Link, Joint, JointType


def evaluate_policy(policy: GaussianPolicy, env: ArmReachEnv, n_episodes: int = 20,
                     obs_noise_std: float = 0.0, action_delay_steps: int = 0, seed: int = 0) -> dict:
    """Runs the policy deterministically (using the mean action, not
    sampled noise - this evaluates what the policy actually WANTS to do,
    not exploration noise) with optional observation noise / action delay
    injected to simulate real-world imperfections."""
    rng = np.random.default_rng(seed)
    final_distances = []

    for ep in range(n_episodes):
        obs = env.reset()
        action_buffer = []  # for simulating control delay
        for step in range(env.max_steps):
            noisy_obs = obs + rng.normal(0, obs_noise_std, size=obs.shape) if obs_noise_std > 0 else obs
            mean_action, _, _, _ = policy.forward(noisy_obs)

            if action_delay_steps > 0:
                action_buffer.append(mean_action)
                action_to_apply = action_buffer[0] if len(action_buffer) <= action_delay_steps else action_buffer.pop(0)
            else:
                action_to_apply = mean_action

            obs, reward, done, info = env.step(action_to_apply)
            if done:
                break
        final_distances.append(info["distance"])

    return {
        "mean_final_distance": float(np.mean(final_distances)),
        "std_final_distance": float(np.std(final_distances)),
    }


def perturbed_config(base_config: ArmConfig, mass_scale: float = 1.0, torque_scale: float = 1.0) -> ArmConfig:
    """Builds a config with mass/torque scaled - standing in for
    manufacturing tolerance, mass estimation error, and motors not
    delivering exactly their rated torque."""
    new_links = [
        Link(name=l.name, length_m=l.length_m, mass_kg=l.mass_kg * mass_scale, radius_m=l.radius_m)
        for l in base_config.links
    ]
    new_joints = [
        Joint(name=j.name, joint_type=j.joint_type, parent_link=j.parent_link, child_link=j.child_link,
              axis=j.axis, lower_limit_rad=j.lower_limit_rad, upper_limit_rad=j.upper_limit_rad,
              max_torque_nm=j.max_torque_nm * torque_scale, max_velocity_rad_s=j.max_velocity_rad_s)
        for j in base_config.joints
    ]
    return ArmConfig(name=base_config.name, links=new_links, joints=new_joints,
                      payload_mass_kg=base_config.payload_mass_kg)


def run_gap_analysis(policy: GaussianPolicy, n_episodes: int = 20, seed: int = 0):
    base_config = default_config()
    results = {}

    print("=== Baseline (exact training conditions) ===")
    env = ArmReachEnv(config=base_config)
    baseline = evaluate_policy(policy, env, n_episodes=n_episodes, seed=seed)
    env.close()
    results["baseline"] = baseline
    print(f"mean_final_distance={baseline['mean_final_distance']:.4f}  std={baseline['std_final_distance']:.4f}")

    print("\n=== Mass perturbation ===")
    for scale in [0.8, 0.9, 1.1, 1.2]:
        config = perturbed_config(base_config, mass_scale=scale)
        env = ArmReachEnv(config=config)
        result = evaluate_policy(policy, env, n_episodes=n_episodes, seed=seed)
        env.close()
        results[f"mass_x{scale}"] = result
        degradation = result["mean_final_distance"] - baseline["mean_final_distance"]
        print(f"mass x{scale}: mean_final_distance={result['mean_final_distance']:.4f}  "
              f"(degradation: {degradation:+.4f})")

    print("\n=== Torque perturbation ===")
    for scale in [0.8, 0.9, 1.1, 1.2]:
        config = perturbed_config(base_config, torque_scale=scale)
        env = ArmReachEnv(config=config)
        result = evaluate_policy(policy, env, n_episodes=n_episodes, seed=seed)
        env.close()
        results[f"torque_x{scale}"] = result
        degradation = result["mean_final_distance"] - baseline["mean_final_distance"]
        print(f"torque x{scale}: mean_final_distance={result['mean_final_distance']:.4f}  "
              f"(degradation: {degradation:+.4f})")

    print("\n=== Observation noise (simulating imperfect encoders) ===")
    for noise_std in [0.01, 0.05, 0.1]:
        env = ArmReachEnv(config=base_config)
        result = evaluate_policy(policy, env, n_episodes=n_episodes, obs_noise_std=noise_std, seed=seed)
        env.close()
        results[f"obs_noise_{noise_std}"] = result
        degradation = result["mean_final_distance"] - baseline["mean_final_distance"]
        print(f"obs_noise_std={noise_std}: mean_final_distance={result['mean_final_distance']:.4f}  "
              f"(degradation: {degradation:+.4f})")

    print("\n=== Action delay (simulating real control loop latency) ===")
    for delay in [1, 3, 5]:
        env = ArmReachEnv(config=base_config)
        result = evaluate_policy(policy, env, n_episodes=n_episodes, action_delay_steps=delay, seed=seed)
        env.close()
        results[f"action_delay_{delay}"] = result
        degradation = result["mean_final_distance"] - baseline["mean_final_distance"]
        print(f"action_delay={delay} steps: mean_final_distance={result['mean_final_distance']:.4f}  "
              f"(degradation: {degradation:+.4f})")

    print("\n" + "=" * 60)
    print("NOTE: this is a robustness/sensitivity proxy, not a real-world")
    print("performance guarantee - no physical hardware was involved.")
    print("=" * 60)

    return results
