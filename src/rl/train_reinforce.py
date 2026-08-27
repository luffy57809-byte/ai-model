"""
REINFORCE training loop for the ArmReachEnv, using GaussianPolicy's
manual gradients (verified correct via numerical gradient checking in
policy.py's design phase).

Algorithm: REINFORCE with reward-to-go (each timestep weighted by the
SUM of rewards from that point forward, not the whole episode's total -
a standard variance-reduction technique, since earlier actions shouldn't
be credited/blamed for rewards that happened before them) and a simple
batch-average baseline (subtract the mean return across the batch, so
the policy only gets pushed toward trajectories that did BETTER than
average, not just toward everything with positive reward).
"""

import numpy as np

from src.rl.arm_env import ArmReachEnv
from src.rl.policy import GaussianPolicy


def collect_episode(env: ArmReachEnv, policy: GaussianPolicy, rng: np.random.Generator):
    obs = env.reset()
    caches, rewards = [], []
    for _ in range(env.max_steps):
        action, cache = policy.sample_action(obs, rng)
        obs, reward, done, info = env.step(action)
        caches.append(cache)
        rewards.append(reward)
        if done:
            break
    final_distance = info["distance"]
    return caches, rewards, final_distance


def compute_reward_to_go(rewards: list, gamma: float = 0.99) -> np.ndarray:
    """returns[t] = sum of (discounted) rewards from timestep t onward -
    NOT the whole episode's total, so each action is only credited for
    what happens after it, not before."""
    returns = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def train(
    n_iterations: int = 100,
    episodes_per_batch: int = 10,
    learning_rate: float = 0.01,
    max_steps: int = 100,
    seed: int = 0,
    std_decay: float = 0.0,
    min_log_std: float = -2.0,
    decay_start_iteration: int = 0,
):
    """
    std_decay: multiplicative decay applied to log_std EVERY iteration,
    independent of the REINFORCE gradient - added because the natural
    gradient signal on log_std was too weak to shrink exploration noise
    on its own (verified: over 300 iterations of real learning, std
    stayed flat around 0.57-0.63 the whole time, capping precision).
    A value like 0.01 means log_std shrinks toward min_log_std by ~1%
    of the remaining gap each iteration. 0.0 disables decay (old
    behavior, kept as the default for backward compatibility).
    """
    env = ArmReachEnv(max_steps=max_steps)
    policy = GaussianPolicy(seed=seed)
    rng = np.random.default_rng(seed)

    history = []

    for iteration in range(n_iterations):
        batch_caches, batch_returns = [], []
        batch_final_distances = []

        for _ in range(episodes_per_batch):
            caches, rewards, final_distance = collect_episode(env, policy, rng)
            returns = compute_reward_to_go(rewards)
            batch_caches.extend(caches)
            batch_returns.extend(returns)
            batch_final_distances.append(final_distance)

        batch_returns = np.array(batch_returns)
        baseline = batch_returns.mean()
        advantages = batch_returns - baseline

        # Accumulate the weighted gradient sum across the whole batch,
        # then average by total timestep count before applying one update.
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

        if std_decay > 0 and iteration >= decay_start_iteration:
            # Explicit decay toward min_log_std, on top of whatever the
            # gradient step already did - forces exploration to shrink
            # over training rather than relying on a weak gradient signal.
            # DELAYED until decay_start_iteration: an earlier attempt with
            # no delay collapsed std to its floor by iteration ~70-80,
            # before the mean action had converged to anything good,
            # trapping the policy in an undertrained local optimum
            # (verified: distance got WORSE, not better, with early decay).
            policy.log_std = policy.log_std - std_decay * (policy.log_std - min_log_std)

        mean_final_distance = float(np.mean(batch_final_distances))
        mean_return = float(batch_returns.mean())
        history.append({"iteration": iteration, "mean_return": mean_return, "mean_final_distance": mean_final_distance})

        if iteration % 10 == 0 or iteration == n_iterations - 1:
            print(f"iter {iteration:4d}  mean_return={mean_return:8.3f}  "
                  f"mean_final_distance={mean_final_distance:.4f}  std={np.exp(policy.log_std).round(3)}")

    env.close()
    return policy, history


if __name__ == "__main__":
    policy, history = train()
