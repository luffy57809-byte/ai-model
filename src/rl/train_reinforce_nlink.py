"""
REINFORCE training for the N-link generalized environment - same
algorithm as train_reinforce.py (reward-to-go + batch-average baseline),
reusing GaussianPolicy AS-IS (its manual backprop math never hardcoded
a specific observation/action dimension, so no changes needed there,
only the sizes passed in).

KEY DIFFERENCE: each episode samples a RANDOM link count (2-5) - this is
what actually tests generalization, as opposed to training on one fixed
arm shape padded to 5 slots (which would prove nothing about
generalizing across different structures).
"""

import numpy as np

from src.rl.arm_env_nlink import NLinkArmReachEnv, OBS_DIM, ACTION_DIM
from src.rl.policy import GaussianPolicy


def collect_episode(env: NLinkArmReachEnv, policy: GaussianPolicy, rng: np.random.Generator):
    obs = env.reset()  # random n_links each episode
    caches, rewards = [], []
    for _ in range(env.max_steps):
        action, cache = policy.sample_action(obs, rng)
        obs, reward, done, info = env.step(action)
        caches.append(cache)
        rewards.append(reward)
        if done:
            break
    return caches, rewards, info["distance"], info["n_links"]


def compute_reward_to_go(rewards: list, gamma: float = 0.99) -> np.ndarray:
    returns = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def train(n_iterations: int = 300, episodes_per_batch: int = 15, learning_rate: float = 0.01,
          max_steps: int = 150, seed: int = 0):
    env = NLinkArmReachEnv(max_steps=max_steps, seed=seed)
    policy = GaussianPolicy(obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=seed)
    rng = np.random.default_rng(seed)

    history = []

    for iteration in range(n_iterations):
        batch_caches, batch_returns = [], []
        batch_distances_by_n_links = {}

        for _ in range(episodes_per_batch):
            caches, rewards, distance, n_links = collect_episode(env, policy, rng)
            returns = compute_reward_to_go(rewards)
            batch_caches.extend(caches)
            batch_returns.extend(returns)
            batch_distances_by_n_links.setdefault(n_links, []).append(distance)

        batch_returns = np.array(batch_returns)
        baseline = batch_returns.mean()
        advantages = batch_returns - baseline

        grad_sum = None
        for cache, advantage in zip(batch_caches, advantages):
            grads = policy.log_prob_grad(cache)
            if grad_sum is None:
                grad_sum = {k: advantage * v for k, v in grads.items()}
            else:
                for k in grad_sum:
                    grad_sum[k] += advantage * grads[k]

        n_timesteps = len(batch_caches)
        grad_avg = {k: v / n_timesteps for k, v in grad_sum.items()}
        policy.apply_gradient_step(grad_avg, learning_rate)

        mean_return = float(batch_returns.mean())
        per_n_links_mean = {n: float(np.mean(d)) for n, d in batch_distances_by_n_links.items()}
        overall_mean_distance = float(np.mean([d for dists in batch_distances_by_n_links.values() for d in dists]))
        history.append({"iteration": iteration, "mean_return": mean_return, "overall_mean_distance": overall_mean_distance})

        if iteration % 10 == 0 or iteration == n_iterations - 1:
            per_n_str = "  ".join(f"n={n}:{d:.3f}" for n, d in sorted(per_n_links_mean.items()))
            print(f"iter {iteration:4d}  mean_return={mean_return:8.3f}  overall_dist={overall_mean_distance:.4f}  [{per_n_str}]")

    env.close()
    return policy, history


if __name__ == "__main__":
    policy, history = train()
