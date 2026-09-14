# This file is part of JEPLO: Joint-Embedding Predictive Learning for LiDAR-Based Legged Locomotion
#
# Copyright (c) 2026 Qihao Yuan
#
# Developer: Qihao Yuan <qihao.yuan@rug.nl>
#
# For commercial use, please contact me at <qihao.yuan@rug.nl> or Kailai Li at <kailai.li@liu.se>.
#
# This file is subject to the terms and conditions outlined in the 'LICENSE' file,
# which is included as part of this source code package.

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
import warnings

import torch
from torch import nn

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation


class ActorCriticTeacher(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_actions,
        env_cfg,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticTeacher.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        activation = resolve_nn_activation("elu")

        self.obs_proprio_range = env_cfg.obs_curr_proprio_range
        self.obs_proprio_hist_range = env_cfg.obs_proprio_hist_range
        self.obs_priv_range = env_cfg.obs_priv_range
        self.obs_height_scan_range = env_cfg.obs_height_scan_range
        self.obs_critic_proprio_range = env_cfg.obs_critic_proprio_range

        self.obs_proprio_dim = env_cfg.obs_proprio_dim
        self.obs_proprio_hist_dim = env_cfg.obs_proprio_hist_dim
        self.obs_priv_dim = env_cfg.obs_priv_dim
        self.obs_height_scan_dim = env_cfg.obs_height_scan_dim
        self.obs_critic_proprio_dim = env_cfg.obs_critic_proprio_dim

        self.num_actions = num_actions
        self.proprio_hist_encode_dim = 64
        self.height_map_encode_dim = 64
        self.priv_encode_dim = 64

        super().__init__(
            num_actor_obs=self.obs_proprio_dim
            + self.proprio_hist_encode_dim
            + self.height_map_encode_dim
            + self.priv_encode_dim,
            num_critic_obs=self.obs_proprio_hist_dim
            + self.obs_priv_dim
            + self.obs_height_scan_dim
            + self.obs_critic_proprio_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation="elu",
            init_noise_std=init_noise_std,
        )

        self.proprio_hist_encoder = nn.Sequential(
            nn.Linear(self.obs_proprio_hist_dim, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, self.proprio_hist_encode_dim),
            nn.Tanh(),
        )

        self.height_map_encoder = nn.Sequential(
            nn.Linear(self.obs_height_scan_dim, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, self.height_map_encode_dim),
            nn.Tanh(),
        )

        self.priv_encoder = nn.Sequential(
            nn.Linear(self.obs_priv_dim, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, self.priv_encode_dim),
            nn.Tanh(),
        )

    def get_policy_params(self):
        params = []
        params.extend(self.actor.parameters())
        params.extend(self.height_map_encoder.parameters())
        params.extend(self.proprio_hist_encoder.parameters())
        params.extend(self.priv_encoder.parameters())
        return params

    def reset(self, dones=None):
        pass

    def act(self, observations, masks=None, hidden_states=None):
        obs = torch.cat(
            [
                observations[..., self.obs_proprio_range[0] : self.obs_proprio_range[1]],
                self.proprio_hist_encoder(
                    observations[..., self.obs_proprio_hist_range[0] : self.obs_proprio_hist_range[1]]
                ),
                self.height_map_encoder(
                    observations[..., self.obs_height_scan_range[0] : self.obs_height_scan_range[1]]
                ),
                self.priv_encoder(observations[..., self.obs_priv_range[0] : self.obs_priv_range[1]]),
            ],
            dim=-1,
        )
        return super().act(obs)

    def act_inference(self, observations):
        obs = torch.cat(
            [
                observations[..., self.obs_proprio_range[0] : self.obs_proprio_range[1]],
                self.proprio_hist_encoder(
                    observations[..., self.obs_proprio_hist_range[0] : self.obs_proprio_hist_range[1]]
                ),
                self.height_map_encoder(
                    observations[..., self.obs_height_scan_range[0] : self.obs_height_scan_range[1]]
                ),
                self.priv_encoder(observations[..., self.obs_priv_range[0] : self.obs_priv_range[1]]),
            ],
            dim=-1,
        )
        return super().act_inference(obs)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        obs = torch.cat(
            [
                critic_observations[..., self.obs_proprio_hist_range[0] : self.obs_proprio_hist_range[1]],
                critic_observations[..., self.obs_priv_range[0] : self.obs_priv_range[1]],
                critic_observations[
                    ..., self.obs_critic_proprio_range[0] : self.obs_critic_proprio_range[1]
                ],
                critic_observations[..., self.obs_height_scan_range[0] : self.obs_height_scan_range[1]],
            ],
            dim=-1,
        )
        return super().evaluate(obs)
