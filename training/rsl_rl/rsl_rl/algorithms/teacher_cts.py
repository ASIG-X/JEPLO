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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules.actor_critic_teacher_cts import ActorCriticTeacherCTS
from rsl_rl.storage.rollout_storage_teacher_cts import RolloutStorageTeacherCTS


class TeacherCTS:
    policy: ActorCriticTeacherCTS

    def __init__(
        self,
        policy,
        storage,
        num_envs,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        student_encoder_learning_rate=2e-4,
        student_encoder_loss_coef=1.0,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        student_env_ratio=0.25,
        warmup_iters=(0, 0),
        normalize_advantage_per_mini_batch=False,
        device="cpu",
        rnd_cfg=None,
        symmetry_cfg=None,
        multi_gpu_cfg=None,
    ):
        if rnd_cfg is not None:
            print("[WARNING] TeacherCTS does not support RND; rnd_cfg will be ignored.")
        if symmetry_cfg is not None:
            print("[WARNING] TeacherCTS does not support symmetry; symmetry_cfg will be ignored.")

        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None
        self.policy = policy
        self.policy.to(self.device)
        self.storage: RolloutStorageTeacherCTS = storage
        self.transition = RolloutStorageTeacherCTS.Transition()

        self.optimizer = optim.Adam(self.policy.ppo_parameters(), lr=learning_rate)
        self.optimizer_student_encoder = optim.Adam(
            self.policy.student_encoder_parameters(), lr=student_encoder_learning_rate
        )

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.student_encoder_loss_coef = student_encoder_loss_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        self.num_envs = num_envs
        self.target_student_num_envs = int(round(num_envs * student_env_ratio))
        self.target_student_num_envs = max(0, min(num_envs - 1, self.target_student_num_envs))
        self.warmup_end_iteration, self.full_cts_iteration = map(int, warmup_iters)
        if not 0 <= self.warmup_end_iteration <= self.full_cts_iteration:
            raise ValueError("warmup_iters must satisfy 0 <= teacher-only end <= full-CTS start.")
        self._all_env_ids = torch.arange(num_envs, device=self.device)
        self.set_iteration(0)

        teacher_ratio = self.teacher_num_envs / num_envs if num_envs > 0 else 0.0
        student_ratio = self.student_num_envs / num_envs if num_envs > 0 else 0.0
        print("=" * 50)
        print("CTS environment split:")
        print(f"  teacher envs: {self.teacher_num_envs} ({teacher_ratio:.1%})")
        print(f"  student envs: {self.student_num_envs} ({student_ratio:.1%})")
        print(f"  schedule: teacher-only until {self.warmup_end_iteration}, full CTS from {self.full_cts_iteration}")
        print("=" * 50)

    def set_iteration(self, iteration):
        if self.storage.step != 0:
            raise RuntimeError("Cannot change the CTS environment split during a rollout.")

        if iteration < self.warmup_end_iteration:
            progress = 0.0
        elif iteration < self.full_cts_iteration:
            progress = (iteration - self.warmup_end_iteration) / (self.full_cts_iteration - self.warmup_end_iteration)
        else:
            progress = 1.0

        self.student_surrogate_weight = progress
        self.student_num_envs = int(round(self.target_student_num_envs * progress))
        self.teacher_num_envs = self.num_envs - self.student_num_envs
        self.student_env_idxs = self._all_env_ids[: self.student_num_envs]
        self.teacher_env_idxs = self._all_env_ids[self.student_num_envs :]
        self.storage.teacher_num_envs = self.teacher_num_envs
        self.storage.student_num_envs = self.student_num_envs

    def log_rollout_schedule(self, iteration):
        print("=" * 50)
        print(f"CTS rollout schedule at iteration {iteration}:")
        print(f"  teacher envs: {self.teacher_num_envs}")
        print(f"  student envs: {self.student_num_envs}")
        print(f"  student surrogate weight: {self.student_surrogate_weight:.3f}")
        print("=" * 50)

    def act(self, obs, high_features):
        ti, si = self.teacher_env_idxs, self.student_env_idxs

        teacher_actions = self.policy.act_teacher(obs[ti])
        teacher_results = (
            teacher_actions.detach(),
            self.policy.evaluate_teacher(obs[ti]).detach(),
            self.policy.get_actions_log_prob(teacher_actions).detach(),
            self.policy.action_mean.detach(),
            self.policy.action_std.detach(),
        )

        if self.student_num_envs > 0:
            student_actions = self.policy.act_student(obs[si], high_features[si])
            student_results = (
                student_actions.detach(),
                self.policy.evaluate_student(obs[si], high_features[si]).detach(),
                self.policy.get_actions_log_prob(student_actions).detach(),
                self.policy.action_mean.detach(),
                self.policy.action_std.detach(),
            )
            results = [torch.cat([x1, x2], dim=0) for x1, x2 in zip(teacher_results, student_results)]
        else:
            results = list(teacher_results)

        self.transition.actions = results[0]
        self.transition.values = results[1]
        self.transition.actions_log_prob = results[2]
        self.transition.action_mean = results[3]
        self.transition.action_sigma = results[4]
        self.transition.observations = torch.cat([obs[ti], obs[si]], dim=0)
        self.transition.high_features = torch.cat([high_features[ti], high_features[si]], dim=0)

        reordered_actions = torch.zeros_like(self.transition.actions)
        reordered_actions[ti] = self.transition.actions[: self.teacher_num_envs]
        if self.student_num_envs > 0:
            reordered_actions[si] = self.transition.actions[self.teacher_num_envs :]
        return reordered_actions

    def process_env_step(self, rewards, dones, infos):
        ti, si = self.teacher_env_idxs, self.student_env_idxs
        rewards = rewards.clone()
        self.transition.rewards = torch.cat([rewards[ti], rewards[si]], dim=0)
        self.transition.dones = torch.cat([dones[ti], dones[si]], dim=0)

        if "time_outs" in infos:
            time_outs = infos["time_outs"].to(self.device)
            reordered_time_outs = torch.cat([time_outs[ti], time_outs[si]], dim=0)
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * reordered_time_outs.unsqueeze(1).to(self.device), 1
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs, high_features):
        ti, si = self.teacher_env_idxs, self.student_env_idxs
        values = [self.policy.evaluate_teacher(obs[ti]).detach()]
        if self.student_num_envs > 0:
            values.append(self.policy.evaluate_student(obs[si], high_features[si]).detach())
        last_values = torch.cat(values, dim=0)
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_student_encoder_loss = 0.0
        mean_student_encoder_head_losses = [0.0] * len(self.policy.teacher_latent_dims)
        num_student_encoder_updates = 0

        teacher_samples = self.teacher_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        student_samples = self.student_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        data = list(self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs))

        for (
            obs_batch,
            high_feature_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in data:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            teacher_actions_log_prob, teacher_value, teacher_mu, teacher_sigma, teacher_entropy = self._evaluate_batch(
                obs_batch[:teacher_samples],
                high_feature_batch[:teacher_samples],
                actions_batch[:teacher_samples],
                is_teacher=True,
            )
            if student_samples > 0:
                student_actions_log_prob, student_value, student_mu, student_sigma, student_entropy = (
                    self._evaluate_batch(
                        obs_batch[teacher_samples:],
                        high_feature_batch[teacher_samples:],
                        actions_batch[teacher_samples:],
                        is_teacher=False,
                    )
                )
                actions_log_prob_batch = torch.cat([teacher_actions_log_prob, student_actions_log_prob], dim=0)
                value_batch = torch.cat([teacher_value, student_value], dim=0)
                mu_batch = torch.cat([teacher_mu, student_mu], dim=0)
                sigma_batch = torch.cat([teacher_sigma, student_sigma], dim=0)
                entropy_batch = torch.cat([teacher_entropy, student_entropy], dim=0)
            else:
                actions_log_prob_batch = teacher_actions_log_prob
                value_batch = teacher_value
                mu_batch = teacher_mu
                sigma_batch = teacher_sigma
                entropy_batch = teacher_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                self._update_learning_rate(old_sigma_batch, old_mu_batch, sigma_batch, mu_batch)

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_losses = torch.max(surrogate, surrogate_clipped)
            surrogate_loss = surrogate_losses[:teacher_samples].mean()
            if student_samples > 0:
                surrogate_loss = (
                    surrogate_loss + self.student_surrogate_weight * surrogate_losses[teacher_samples:].mean()
                )

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters(self.policy.ppo_parameters())
            nn.utils.clip_grad_norm_(self.policy.ppo_parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        if self.student_encoder_loss_coef > 0.0:
            for (
                obs_batch,
                high_feature_batch,
                *_,
            ) in data:
                if self.student_surrogate_weight < 1.0 or student_samples == 0:
                    student_obs = obs_batch
                    student_high_features = high_feature_batch
                else:
                    student_obs = obs_batch[teacher_samples:]
                    student_high_features = high_feature_batch[teacher_samples:]
                student_latent = self.policy.encode_student_latent(student_obs, student_high_features)
                with torch.no_grad():
                    teacher_latent = self.policy.encode_teacher_latent(student_obs).detach()
                student_chunks = torch.split(student_latent, self.policy.teacher_latent_dims, dim=-1)
                teacher_chunks = torch.split(teacher_latent, self.policy.teacher_latent_dims, dim=-1)
                head_losses = [
                    (teacher_chunk - student_chunk).pow(2).mean()
                    for teacher_chunk, student_chunk in zip(teacher_chunks, student_chunks)
                ]
                student_encoder_loss = torch.stack(head_losses).mean()
                loss = self.student_encoder_loss_coef * student_encoder_loss

                self.optimizer_student_encoder.zero_grad()
                loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters(self.policy.student_encoder_parameters())
                nn.utils.clip_grad_norm_(self.policy.student_encoder_parameters(), self.max_grad_norm)
                self.optimizer_student_encoder.step()

                mean_student_encoder_loss += student_encoder_loss.item()
                for i, head_loss in enumerate(head_losses):
                    mean_student_encoder_head_losses[i] += head_loss.item()
                num_student_encoder_updates += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if num_student_encoder_updates > 0:
            mean_student_encoder_loss /= num_student_encoder_updates
            mean_student_encoder_head_losses = [
                loss / num_student_encoder_updates for loss in mean_student_encoder_head_losses
            ]

        self.storage.clear()
        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "student_encoder": mean_student_encoder_loss,
            "student_encoder_history": mean_student_encoder_head_losses[0],
            "student_encoder_height": mean_student_encoder_head_losses[1],
            "student_encoder_privileged": mean_student_encoder_head_losses[2],
            "student_rollout_ratio": self.student_num_envs / self.num_envs,
            "student_surrogate_weight": self.student_surrogate_weight,
        }

    def _evaluate_batch(self, obs, high_features, actions, is_teacher):
        if is_teacher:
            self.policy.act_teacher(obs)
            value = self.policy.evaluate_teacher(obs)
        else:
            self.policy.act_student(obs, high_features)
            value = self.policy.evaluate_student(obs, high_features)
        return (
            self.policy.get_actions_log_prob(actions),
            value,
            self.policy.action_mean,
            self.policy.action_std,
            self.policy.entropy,
        )

    def _update_learning_rate(self, old_sigma_batch, old_mu_batch, sigma_batch, mu_batch):
        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                / (2.0 * torch.square(sigma_batch))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)

            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size

            if self.gpu_global_rank == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            if self.is_multi_gpu:
                lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(lr_tensor, src=0)
                self.learning_rate = lr_tensor.item()

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def broadcast_parameters(self):
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self, parameters):
        parameters = list(parameters)
        grads = [param.grad.view(-1) for param in parameters if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in parameters:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
