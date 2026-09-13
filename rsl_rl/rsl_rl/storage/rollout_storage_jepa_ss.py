# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


class RolloutStorageJepaSS:
    class Transition:
        def __init__(self):
            self.proprio_hist = None
            self.depth_stack = None
            self.action_hist = None
            self.hidden_states = None  # GRU hidden BEFORE the forward pass
            self.dones = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        prop_hist_shape,
        depth_stack_shape,
        action_hist_shape,
        gru_hidden_dim,
        device="cpu",
    ):
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.proprio_hist = torch.zeros(num_transitions_per_env, num_envs, *prop_hist_shape, device=device)
        self.depth_stack = torch.zeros(num_transitions_per_env, num_envs, *depth_stack_shape, device=device)
        self.action_hist = torch.zeros(num_transitions_per_env, num_envs, *action_hist_shape, device=device)

        # Episode boundaries
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()

        # GRU hidden states (stored BEFORE each forward pass)
        self.gru_hidden_dim = gru_hidden_dim
        self.hidden_states = torch.zeros(num_transitions_per_env, num_envs, gru_hidden_dim, device=device)

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! Call clear() before adding new transitions.")

        self.proprio_hist[self.step].copy_(transition.proprio_hist)
        self.depth_stack[self.step].copy_(transition.depth_stack)
        self.action_hist[self.step].copy_(transition.action_hist)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.hidden_states[self.step].copy_(transition.hidden_states)

        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """Generate mini-batches by slicing envs into contiguous chunks.

        Each batch contains full time sequences for a subset of envs.
        Hidden state continuity and done-based resets are handled in train_step.

        Yields dicts with keys:
            prop_hist:             (T, batch_envs, prop_hist_dim)
            depth_stack:           (T, batch_envs, num_depth_channels, H, W)
            action_hist:           (T, batch_envs, action_dim)
            dones:                 (T, batch_envs, 1)
            hidden_states:         (T, batch_envs, gru_hidden_dim)
        """
        T = self.step  # actual number of transitions added
        mini_batch_size = self.num_envs // num_mini_batches
        mini_batch_size = min(mini_batch_size, 256)
        num_epochs = num_epochs
        for ep in range(num_epochs):
            perm = torch.randperm(self.num_envs, device=self.device)
            for i in range(num_mini_batches):
                idx = perm[i * mini_batch_size : (i + 1) * mini_batch_size]

                yield {
                    "prop_hist": self.proprio_hist[:T, idx],
                    "depth_stack": self.depth_stack[:T, idx],
                    "action_hist": self.action_hist[:T, idx],
                    "dones": self.dones[:T, idx],
                    "hidden_states": self.hidden_states[:T, idx],
                }

                # only randomly sample a single mini-batch per epoch
                break
