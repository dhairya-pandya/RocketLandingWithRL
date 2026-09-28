"""OpenAI-style evolution strategies (Salimans et al., 2017): gradient-free search over the
weights of a small deterministic policy, using antithetic noise and rank-shaped fitness."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from rocketlander.common.actor_critic import ACT_SIZE
from rocketlander.common.logger import Logger
from rocketlander.common.normalization import RunningMeanStd
from rocketlander.envs.rocket_env import OBS_SIZE, RocketLanderEnv
from rocketlander.evaluation import EVAL_SEEDS, eval_metrics


@dataclass
class ESConfig:
    total_steps: int = 20_000_000
    population: int = 64  # perturbations per generation (half are mirror images)
    episodes_per_member: int = 2  # every member sees the same start states
    sigma: float = 0.05  # perturbation scale
    learning_rate: float = 0.02
    weight_decay: float = 0.005
    hidden: list[int] = field(default_factory=lambda: [32, 32])
    eval_interval: int = 1_000_000
    eval_episodes: int = 20


class ESAgent(nn.Module):
    """Deterministic tanh MLP policy; its weights are one flat vector for the search."""

    def __init__(self, hidden: list[int]) -> None:
        super().__init__()
        self.hidden = list(hidden)
        sizes = [OBS_SIZE, *self.hidden, ACT_SIZE]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:], strict=False))
        self.obs_rms = RunningMeanStd((OBS_SIZE,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        x = torch.as_tensor(self.obs_rms.normalize(obs), dtype=torch.float32)
        return self(x).numpy().astype(np.float32)

    def get_flat(self) -> np.ndarray:
        return nn.utils.parameters_to_vector(self.parameters()).detach().numpy().copy()

    def set_flat(self, flat: np.ndarray) -> None:
        nn.utils.vector_to_parameters(torch.as_tensor(flat, dtype=torch.float32), self.parameters())

    def batched_policy(self, flats: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Actions for many parameter vectors at once: flats (P, D), obs (P, OBS_SIZE)."""
        x = self.obs_rms.normalize(obs)
        offset = 0
        for layer in self.layers:
            n_out, n_in = layer.weight.shape
            w = flats[:, offset : offset + n_out * n_in].reshape(-1, n_out, n_in)
            offset += n_out * n_in
            b = flats[:, offset : offset + n_out]
            offset += n_out
            x = np.tanh(np.einsum("poi,pi->po", w, x) + b)
        return x.astype(np.float32)

    def architecture(self) -> dict:
        return {"hidden": self.hidden}

    def extra_state(self) -> dict:
        return {"obs_rms": self.obs_rms.state_dict()}

    def load_extra_state(self, state: dict) -> None:
        self.obs_rms.load_state_dict(state["obs_rms"])


def centered_ranks(values: np.ndarray) -> np.ndarray:
    """Map fitness to evenly spaced values in [-0.5, 0.5]; robust to outliers and reward scale."""
    ranks = np.empty(len(values))
    ranks[values.argsort()] = np.arange(len(values))
    return ranks / (len(values) - 1) - 0.5


def evaluate_population(agent, flats, level, start_seeds) -> tuple[np.ndarray, int, np.ndarray]:
    """Mean shaped return of each parameter vector over the same start seeds, all in lockstep."""
    members, episodes = len(flats), len(start_seeds)
    envs = [RocketLanderEnv(level=level) for _ in range(members * episodes)]
    obs = np.stack([env.reset(seed=start_seeds[i % episodes])[0] for i, env in enumerate(envs)])
    params = np.repeat(flats, episodes, axis=0)
    returns = np.zeros(len(envs))
    active = np.ones(len(envs), dtype=bool)
    steps, visited = 0, []
    while active.any():
        idx = np.flatnonzero(active)
        actions = agent.batched_policy(params[idx], obs[idx])
        visited.append(obs[idx[:: max(1, len(idx) // 8)]])
        for j, i in enumerate(idx):
            obs[i], reward, terminated, truncated, _ = envs[i].step(actions[j])
            returns[i] += reward
            if terminated or truncated:
                active[i] = False
        steps += len(idx)
    return returns.reshape(members, episodes).mean(axis=1), steps, np.concatenate(visited)


def train_es(
    config: ESConfig, level: str, seed: int, logger: Logger, agent: ESAgent | None = None
) -> ESAgent:
    torch.manual_seed(seed)
    agent = agent or ESAgent(config.hidden)
    rng = np.random.default_rng(seed)
    theta = agent.get_flat()
    optimizer_m, optimizer_v = np.zeros_like(theta), np.zeros_like(theta)  # Adam moments
    half = config.population // 2
    global_step, next_eval, generation, start = 0, config.eval_interval, 0, time.time()
    while global_step < config.total_steps:
        generation += 1
        noise = rng.standard_normal((half, len(theta)))
        flats = np.concatenate([theta + config.sigma * noise, theta - config.sigma * noise])
        start_seeds = rng.integers(0, min(EVAL_SEEDS), size=config.episodes_per_member).tolist()
        fitness, steps, visited = evaluate_population(agent, flats, level, start_seeds)
        global_step += steps
        agent.obs_rms.update(visited)

        ranks = centered_ranks(fitness)
        gradient = (ranks[:half] - ranks[half:]) @ noise / (config.population * config.sigma)
        gradient -= config.weight_decay * theta
        optimizer_m = 0.9 * optimizer_m + 0.1 * gradient
        optimizer_v = 0.999 * optimizer_v + 0.001 * gradient**2
        m_hat = optimizer_m / (1 - 0.9**generation)
        v_hat = optimizer_v / (1 - 0.999**generation)
        theta = theta + config.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)  # ascent
        agent.set_flat(theta)

        metrics = {
            "train/fitness_mean": float(fitness.mean()),
            "train/fitness_max": float(fitness.max()),
            "train/steps_per_second": global_step / (time.time() - start),
        }
        if global_step >= next_eval or global_step >= config.total_steps:
            metrics |= eval_metrics(agent, level, config.eval_episodes)
            next_eval += config.eval_interval
        logger.log(global_step, metrics)
    return agent
