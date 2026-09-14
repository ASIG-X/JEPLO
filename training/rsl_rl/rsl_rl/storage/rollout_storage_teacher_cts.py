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

from __future__ import annotations

from functools import partial

import torch


class RolloutStorageTeacherCTS:
    class Transition:
        def __init__(self):
            self.observations = None
            self.high_features = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        teacher_num_envs,
        num_transitions_per_env,
        obs_shape,
        high_feature_shape,
        actions_shape,
        device="cpu",
    ):
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.teacher_num_envs = teacher_num_envs
        self.student_num_envs = num_envs - teacher_num_envs

        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=device)
        self.high_features = torch.zeros(num_transitions_per_env, num_envs, *high_feature_shape, device=device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()

        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! Call clear() before adding new transitions.")

        self.observations[self.step].copy_(transition.observations)
        self.high_features[self.step].copy_(transition.high_features)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage=True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        teacher_samples_num = self.teacher_num_envs * self.num_transitions_per_env
        student_samples_num = self.student_num_envs * self.num_transitions_per_env
        teacher_mini_batch_size = teacher_samples_num // num_mini_batches
        student_mini_batch_size = student_samples_num // num_mini_batches if self.student_num_envs > 0 else 0

        teacher_indices = torch.randperm(teacher_samples_num, requires_grad=False, device=self.device)
        if student_samples_num > 0:
            student_indices = teacher_samples_num + torch.randperm(
                student_samples_num, requires_grad=False, device=self.device
            )
        else:
            student_indices = torch.empty(0, dtype=torch.long, device=self.device)

        observations = self.observations.transpose(0, 1).flatten(0, 1)
        high_features = self.high_features.transpose(0, 1).flatten(0, 1)
        actions = self.actions.transpose(0, 1).flatten(0, 1)
        values = self.values.transpose(0, 1).flatten(0, 1)
        returns = self.returns.transpose(0, 1).flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.transpose(0, 1).flatten(0, 1)
        advantages = self.advantages.transpose(0, 1).flatten(0, 1)
        old_mu = self.mu.transpose(0, 1).flatten(0, 1)
        old_sigma = self.sigma.transpose(0, 1).flatten(0, 1)

        def _get_teacher_student_samples(data, batch_slice):
            (i1, i2), (j1, j2) = batch_slice
            chunks = [data[teacher_indices[i1:i2]]]
            if student_mini_batch_size > 0:
                chunks.append(data[student_indices[j1:j2]])
            return torch.cat(chunks, dim=0).detach()

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                batch_slice = (
                    (i * teacher_mini_batch_size, (i + 1) * teacher_mini_batch_size),
                    (i * student_mini_batch_size, (i + 1) * student_mini_batch_size),
                )
                get_batch = partial(_get_teacher_student_samples, batch_slice=batch_slice)
                yield (*map(
                    get_batch,
                    [
                        observations,
                        high_features,
                        actions,
                        values,
                        advantages,
                        returns,
                        old_actions_log_prob,
                        old_mu,
                        old_sigma,
                    ],
                ),)
