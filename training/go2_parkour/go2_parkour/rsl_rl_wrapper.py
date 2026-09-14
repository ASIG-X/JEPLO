"""Compatibility wrapper for the project's RSL-RL runners."""

from __future__ import annotations

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper as IsaacLabRslRlVecEnvWrapper


class RslRlVecEnvWrapper(IsaacLabRslRlVecEnvWrapper):
    """Expose the Isaac Lab 2.1 observation contract on Isaac Lab 2.3."""

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        observations = super().get_observations().to_dict()
        return observations["policy"], {"observations": observations}

    def reset(self) -> tuple[torch.Tensor, dict]:
        observations, _ = super().reset()
        observations = observations.to_dict()
        return observations["policy"], {"observations": observations}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        observations, rewards, dones, extras = super().step(actions)
        observations = observations.to_dict()
        extras["observations"] = observations
        return observations["policy"], rewards, dones, extras
