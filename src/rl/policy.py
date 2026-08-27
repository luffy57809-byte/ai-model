"""
A small Gaussian MLP policy (7 -> 16 -> 2) with manual backpropagation -
no PyTorch/autodiff, matching this project's NumPy-only approach for the
RL module.

Architecture: obs -> tanh(hidden) -> tanh(mean), with a separate learned
log_std (state-independent) controlling exploration noise. Actions are
sampled from Normal(mean, std) and clipped to [-1, 1] by the environment.

Manual gradient: computes d(log_prob)/d(params) analytically for REINFORCE
- this is genuine backpropagation through the two-layer network, derived
by hand rather than obtained from an autodiff library.
"""

import numpy as np


class GaussianPolicy:
    def __init__(self, obs_dim: int = 7, hidden_dim: int = 16, action_dim: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Small random init, scaled by fan-in (standard practice to keep
        # initial activations well-behaved).
        self.W1 = rng.normal(0, 1.0 / np.sqrt(obs_dim), size=(obs_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(hidden_dim), size=(hidden_dim, action_dim))
        self.b2 = np.zeros(action_dim)
        self.log_std = np.full(action_dim, -0.5)  # std ~ 0.6 initially

    def forward(self, obs: np.ndarray):
        """Returns (mean, hidden_pre_activation, hidden_post_activation) -
        intermediates are needed for the manual backward pass."""
        h_pre = obs @ self.W1 + self.b1
        h = np.tanh(h_pre)
        mean_pre = h @ self.W2 + self.b2
        mean = np.tanh(mean_pre)
        return mean, h_pre, h, mean_pre

    def sample_action(self, obs: np.ndarray, rng: np.random.Generator):
        mean, h_pre, h, mean_pre = self.forward(obs)
        std = np.exp(self.log_std)
        noise = rng.normal(0, 1, size=mean.shape)
        action = mean + std * noise
        cache = {"obs": obs, "h_pre": h_pre, "h": h, "mean_pre": mean_pre, "mean": mean, "action": action}
        return action, cache

    def log_prob_grad(self, cache: dict) -> dict:
        """
        Computes d(log_prob)/d(each parameter) for the sampled action in
        cache, via manual backpropagation. This is the REINFORCE gradient
        direction for a single timestep - the training loop scales each
        timestep's gradient by (return - baseline) and sums/averages
        across a batch before applying an update.
        """
        obs, h_pre, h, mean_pre, mean, action = (
            cache["obs"], cache["h_pre"], cache["h"], cache["mean_pre"], cache["mean"], cache["action"]
        )
        std = np.exp(self.log_std)

        # d(log_prob)/d(mean) for a diagonal Gaussian: (action - mean) / std^2
        d_log_prob_d_mean = (action - mean) / (std ** 2)
        # d(log_prob)/d(log_std): standard Gaussian log-prob derivative wrt log_std
        d_log_prob_d_log_std = ((action - mean) ** 2) / (std ** 2) - 1.0

        # Backprop through mean = tanh(mean_pre)
        d_mean_pre = d_log_prob_d_mean * (1 - mean ** 2)

        # Backprop through mean_pre = h @ W2 + b2
        d_W2 = np.outer(h, d_mean_pre)
        d_b2 = d_mean_pre
        d_h = d_mean_pre @ self.W2.T

        # Backprop through h = tanh(h_pre)
        d_h_pre = d_h * (1 - h ** 2)

        # Backprop through h_pre = obs @ W1 + b1
        d_W1 = np.outer(obs, d_h_pre)
        d_b1 = d_h_pre

        return {
            "W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2,
            "log_std": d_log_prob_d_log_std,
        }

    def apply_gradient_step(self, grads: dict, learning_rate: float):
        """Gradient ASCENT (maximizing expected return), since grads here
        are d(log_prob)/d(params) scaled by advantage - REINFORCE's update
        direction, not a loss to minimize."""
        self.W1 += learning_rate * grads["W1"]
        self.b1 += learning_rate * grads["b1"]
        self.W2 += learning_rate * grads["W2"]
        self.b2 += learning_rate * grads["b2"]
        self.log_std += learning_rate * grads["log_std"]
        # Keep std within a sane range - unconstrained log_std can drift
        # to produce either near-zero exploration or huge, useless noise.
        self.log_std = np.clip(self.log_std, -2.0, 1.0)
