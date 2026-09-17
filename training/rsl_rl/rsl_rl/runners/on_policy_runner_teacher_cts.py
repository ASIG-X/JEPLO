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

import json
import os
import shutil
import statistics
import time
from collections import deque
from dataclasses import asdict

import numpy as np
import torch

import rsl_rl
import wandb
from rsl_rl.algorithms.teacher_cts import TeacherCTS
from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.actor_critic_teacher_cts import ActorCriticTeacherCTS
from rsl_rl.modules.jepa_estimator_sensor_ss import JepaEstimatorSensorSS
from rsl_rl.runners.on_policy_runner_teacher import OnPolicyRunnerTeacher
from rsl_rl.storage.rollout_storage_jepa_ss import RolloutStorageJepaSS
from rsl_rl.storage.rollout_storage_teacher_cts import RolloutStorageTeacherCTS
from rsl_rl.utils import store_code_state


class OnPolicyRunnerTeacherCTS(OnPolicyRunnerTeacher):
    """Concurrent teacher-student runner for the teacher policy pipeline."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device="cpu",
        wandb_project: str | None = None,
    ):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.wandb_project = wandb_project
        self.launch_command = self.cfg.get("launch_command")
        self.launch_cwd = self.cfg.get("launch_cwd")
        self.wandb_run_name = self.cfg.get("wandb_run_name")

        self._configure_multi_gpu()

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        observations = extras.get("observations", {})
        if "depth" not in observations:
            raise RuntimeError("CTS training requires depth observations.")

        num_privileged_obs = num_obs
        print("on policy runner CTS:")
        print(f"num_obs: {num_obs}, num_privileged_obs: {num_privileged_obs}")

        self._num_obs_hist = self.env.cfg.num_obs_hist
        self._student_proprio_dim = 45
        self._student_proprio_hist_dim = self._student_proprio_dim * self._num_obs_hist
        for k, v in self.env.cfg.to_dict().items():
            if k.startswith("obs_"):
                setattr(self, "_" + k, v)
                print(f"setting attribute {k} to {v} from env config")

        self._build_jepa_estimator()
        self._depth_shape = tuple(observations["depth"].shape[1:])
        print(f"Depth observation shape: {self._depth_shape}")

        self.policy_cfg.pop("class_name", None)
        print(f"Using policy class: {ActorCriticTeacherCTS}")
        policy = ActorCriticTeacherCTS(
            num_actions=self.env.num_actions,
            env_cfg=self.env.cfg,
            jepa_feature_dim=self._sensor_latent_dim,
            **self.policy_cfg,
        ).to(self.device)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.onnx_test_samples = int(self.cfg.get("onnx_test_samples", 10))
        if self.onnx_test_samples < 1:
            raise ValueError("onnx_test_samples must be positive.")

        self.alg_cfg.pop("class_name", None)
        student_env_ratio = float(self.alg_cfg.get("student_env_ratio", 0.25))
        teacher_num_envs = self._get_teacher_num_envs(self.env.num_envs, student_env_ratio)
        storage = RolloutStorageTeacherCTS(
            num_envs=self.env.num_envs,
            teacher_num_envs=teacher_num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            obs_shape=[num_obs],
            high_feature_shape=[self._sensor_latent_dim],
            actions_shape=[self.env.num_actions],
            device=self.device,
        )

        self.alg = TeacherCTS(
            policy,
            storage=storage,
            num_envs=self.env.num_envs,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )
        print(f"Using algorithm class: {TeacherCTS}")
        self._log_cts_environment_division()

        self._init_estimator_storage()

        self.cfg["empirical_normalization"] = False
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        if self.wandb_project is not None:
            wandb.login()

    def _build_jepa_estimator(self):
        print("Building JEPA sensor estimator with the following configuration:")
        self._estimator_update_interval = 5
        self._estimator = JepaEstimatorSensorSS(
            temporal_steps=self._num_obs_hist,
            num_one_step_obs=self._student_proprio_dim,
            prop_hist_dim=self._student_proprio_hist_dim,
            num_depth_channels=2,
            action_hist_dim=self.env.num_actions * self._estimator_update_interval,
        ).to(self.device)
        self._sensor_latent_dim = self._estimator.get_feature_dim()
        print(f"JEPA sensor estimator latent dim: {self._sensor_latent_dim}")
        print(f"JEPA sensor estimator has {sum(p.numel() for p in self._estimator.parameters())} parameters")

    def _init_estimator_storage(self):
        self.estimator_storage = RolloutStorageJepaSS(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env // self._estimator_update_interval + 1,
            prop_hist_shape=(self._estimator.prop_hist_dim,),
            depth_stack_shape=self._estimator.get_depth_shape(),
            action_hist_shape=(self.env.num_actions * self._estimator_update_interval,),
            gru_hidden_dim=self._estimator.get_hidden_dim(),
            device=self.device,
        )
        self.estimator_transition = RolloutStorageJepaSS.Transition()
        self._num_mini_batches_jepa = 2
        self._num_learning_epochs_jepa = 5

    def _log_cts_environment_division(self):
        total_envs = self.env.num_envs
        teacher_ids = self.alg.teacher_env_idxs.detach().cpu().numpy()
        student_ids = self.alg.student_env_idxs.detach().cpu().numpy()
        teacher_num_envs = len(teacher_ids)
        student_num_envs = len(student_ids)

        print("=" * 50)
        print("CTS environment division summary:")
        print(f"  total envs: {total_envs}")
        print(f"  teacher envs: {teacher_num_envs} ({teacher_num_envs / total_envs:.1%})")
        print(f"  student envs: {student_num_envs} ({student_num_envs / total_envs:.1%})")

        env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(env, "_env_terrain_names"):
            print("  subterrain split: unavailable")
            print("=" * 50)
            return

        terrain_names = np.asarray(env._env_terrain_names)
        if terrain_names.ndim > 1:
            terrain_names = terrain_names[:, -1]

        terrain_order = []
        terrain_cfg = getattr(getattr(env.cfg, "terrain", None), "terrain_generator", None)
        if terrain_cfg is not None:
            terrain_order = list(terrain_cfg.sub_terrains.keys())
        terrain_order += [name for name in np.unique(terrain_names) if name not in terrain_order]

        print("  subterrain split:")
        print("    terrain | teacher envs | student envs")
        for terrain_name in terrain_order:
            teacher_count = int(np.count_nonzero(terrain_names[teacher_ids] == terrain_name))
            student_count = int(np.count_nonzero(terrain_names[student_ids] == terrain_name))
            teacher_ratio = teacher_count / teacher_num_envs if teacher_num_envs > 0 else 0.0
            student_ratio = student_count / student_num_envs if student_num_envs > 0 else 0.0
            print(f"    {terrain_name}: {teacher_count} ({teacher_ratio:.1%}) | {student_count} ({student_ratio:.1%})")
        print("=" * 50)

    def learn(self, num_learning_iterations: int):  # noqa: C901
        if self.wandb_project is None:
            raise ValueError("wandb_project must be specified for logging.")

        run = wandb.init(project=self.wandb_project, name=self.wandb_run_name)
        wandb.config.update({"runner_cfg": self.cfg})
        wandb.config.update({"policy_cfg": self.policy_cfg})
        wandb.config.update({"alg_cfg": self.alg_cfg})

        try:
            wandb.config.update({"env_cfg": self.env.cfg.to_dict()})
        except Exception:
            wandb.config.update({"env_cfg": asdict(self.env.cfg)})

        obs, extras = self.env.get_observations()
        depth_stack = extras["observations"]["depth"].to(self.device)
        obs = obs.to(self.device)
        self.train_mode()

        ep_infos = []
        teacher_rewbuffer = deque(maxlen=100)
        teacher_lenbuffer = deque(maxlen=100)
        student_rewbuffer = deque(maxlen=100)
        student_lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        estimator_hidden = torch.zeros(self.env.num_envs, self._estimator.get_hidden_dim(), device=self.device)
        sensor_latent = torch.zeros(self.env.num_envs, self._sensor_latent_dim, device=self.device)
        est_proprio_hist = self.alg.policy.extract_student_proprio_hist(obs).clone()
        est_depth_stack = depth_stack.clone()
        est_action_hist = torch.zeros(
            self.env.num_envs, self._estimator_update_interval, self.env.num_actions, device=self.device
        )
        estimator_interval_dones = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
        estimator_has_pending_transition = False

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            estimator_params = [self._estimator.state_dict()]
            torch.distributed.broadcast_object_list(estimator_params, src=0)
            self._estimator.load_state_dict(estimator_params[0])

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            self.alg.set_iteration(it)
            is_schedule_boundary = it == start_iter or it in (
                self.alg.warmup_end_iteration,
                self.alg.full_cts_iteration,
            )
            is_ramp_summary = (
                self.alg.warmup_end_iteration < it < self.alg.full_cts_iteration
                and (it - self.alg.warmup_end_iteration) % 500 == 0
            )
            if is_schedule_boundary or is_ramp_summary:
                self.alg.log_rollout_schedule(it)
            if it % 20 == 0 or is_schedule_boundary or is_ramp_summary:
                self._log_cts_environment_division()
            start = time.time()

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    estimator_collect = self.env.unwrapped.common_step_counter % self._estimator_update_interval == 0
                    if estimator_collect:
                        if estimator_has_pending_transition:
                            self.process_estimator_step(estimator_interval_dones, est_action_hist.flatten(1))
                        estimator_interval_dones[:] = False
                        est_action_hist.zero_()

                        self.record_estimator_data(
                            prop_hist=est_proprio_hist,
                            depth_stack=est_depth_stack,
                            hidden_states=estimator_hidden,
                        )

                        estimator_feature, estimator_hidden = self._estimator(
                            prop_hist=est_proprio_hist,
                            depth_stack=est_depth_stack,
                            hidden=estimator_hidden,
                        )
                        sensor_latent = estimator_feature
                        estimator_has_pending_transition = True

                    actions = self.alg.act(obs, sensor_latent)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    depth_stack = infos["observations"]["depth"].to(self.device)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    obs = self.obs_normalizer(obs)

                    done_mask = dones.bool()
                    est_action_hist = torch.cat([est_action_hist[:, 1:], actions.unsqueeze(1)], dim=1)
                    estimator_interval_dones |= done_mask

                    estimator_hidden[done_mask] = 0
                    sensor_latent[done_mask] = 0
                    est_action_hist[done_mask] = 0

                    est_depth_stack[:] = depth_stack
                    est_proprio_hist[:] = self.alg.policy.extract_student_proprio_hist(obs)

                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])

                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                        if done_ids.numel() > 0:
                            teacher_done_mask = torch.isin(done_ids, self.alg.teacher_env_idxs)
                            teacher_done_ids = done_ids[teacher_done_mask]
                            student_done_ids = done_ids[~teacher_done_mask]

                            teacher_rewbuffer.extend(cur_reward_sum[teacher_done_ids].cpu().numpy().tolist())
                            teacher_lenbuffer.extend(cur_episode_length[teacher_done_ids].cpu().numpy().tolist())
                            student_rewbuffer.extend(cur_reward_sum[student_done_ids].cpu().numpy().tolist())
                            student_lenbuffer.extend(cur_episode_length[student_done_ids].cpu().numpy().tolist())
                            cur_reward_sum[done_ids] = 0
                            cur_episode_length[done_ids] = 0

                if estimator_has_pending_transition:
                    self.process_estimator_step(estimator_interval_dones, est_action_hist.flatten(1))
                    estimator_has_pending_transition = False
                    estimator_interval_dones[:] = False
                    est_action_hist.zero_()

                stop = time.time()
                collection_time = stop - start
                start = stop

                self.alg.compute_returns(obs, sensor_latent)

            loss_dict = self.alg.update()
            loss_dict.update(self.update_estimator())

            stop = time.time()
            learn_time = stop - start
            # Keep this counter as the next iteration to run.  Checkpoints are
            # written after an update, so resuming from iteration ``it`` must
            # continue at ``it + 1`` instead of collecting that rollout twice.
            self.current_learning_iteration = it + 1
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals(), wandb_run=run)

            if it % self.save_interval == 0:
                self._save_checkpoint_and_export(f"model_{it}.pt", obs, depth_stack)

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(
                    self.log_dir,
                    self.git_status_repos,
                    include_recent_history=True,
                    launch_command=self.launch_command,
                    launch_cwd=self.launch_cwd,
                )

        if self.current_learning_iteration > 0:
            last_completed_iteration = self.current_learning_iteration - 1
            self._save_checkpoint_and_export(f"model_{last_completed_iteration}.pt", obs, depth_stack)

        run.finish()

    def log(self, locs: dict, width: int = 80, pad: int = 35, wandb_run=None):
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    wandb_run.log({key: value}, step=locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    wandb_run.log({"Episode/" + key: value}, step=locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / iteration_time)

        for key, value in locs["loss_dict"].items():
            wandb_run.log({f"Loss/{key}": value}, step=locs["it"])
        wandb_run.log({"Loss/learning_rate": self.alg.learning_rate}, step=locs["it"])
        wandb_run.log({"Policy/mean_noise_std": mean_std.item()}, step=locs["it"])
        wandb_run.log({"Perf/total_fps": fps}, step=locs["it"])
        wandb_run.log({"Perf/collection time": locs["collection_time"]}, step=locs["it"])
        wandb_run.log({"Perf/learning_time": locs["learn_time"]}, step=locs["it"])

        teacher_rewbuffer = locs["teacher_rewbuffer"]
        teacher_lenbuffer = locs["teacher_lenbuffer"]
        student_rewbuffer = locs["student_rewbuffer"]
        student_lenbuffer = locs["student_lenbuffer"]
        if teacher_rewbuffer:
            wandb_run.log({"Train/mean_teacher_reward": statistics.mean(teacher_rewbuffer)}, step=locs["it"])
            wandb_run.log(
                {"Train/mean_teacher_episode_length": statistics.mean(teacher_lenbuffer)}, step=locs["it"]
            )
        if student_rewbuffer:
            wandb_run.log({"Train/mean_student_reward": statistics.mean(student_rewbuffer)}, step=locs["it"])
            wandb_run.log(
                {"Train/mean_student_episode_length": statistics.mean(student_lenbuffer)}, step=locs["it"]
            )

        iteration_header = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "
        log_string = (
            f"""{"#" * width}\n"""
            f"""{iteration_header.center(width, " ")}\n\n"""
            f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {locs["learn_time"]:.3f}s)\n"""
            f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
        )
        for key, value in locs["loss_dict"].items():
            log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""

        if teacher_rewbuffer:
            log_string += f"""{"Mean teacher reward:":>{pad}} {statistics.mean(teacher_rewbuffer):.2f}\n"""
            log_string += (
                f"""{"Mean teacher episode length:":>{pad}} {statistics.mean(teacher_lenbuffer):.2f}\n"""
            )
        if student_rewbuffer:
            log_string += f"""{"Mean student reward:":>{pad}} {statistics.mean(student_rewbuffer):.2f}\n"""
            log_string += (
                f"""{"Mean student episode length:":>{pad}} {statistics.mean(student_lenbuffer):.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {
                time.strftime(
                    "%H:%M:%S",
                    time.gmtime(
                        self.tot_time
                        / (locs["it"] - locs["start_iter"] + 1)
                        * (locs["start_iter"] + locs["num_learning_iterations"] - locs["it"])
                    ),
                )
            }\n"""
        )
        print(log_string)

    def record_estimator_data(self, prop_hist, depth_stack, hidden_states):
        self.estimator_transition.proprio_hist = prop_hist.detach()
        self.estimator_transition.depth_stack = depth_stack.detach()
        self.estimator_transition.hidden_states = hidden_states.detach()

    def process_estimator_step(self, dones, action_hist):
        self.estimator_transition.action_hist = action_hist.detach()
        self.estimator_transition.dones = dones.detach()
        self.estimator_storage.add_transitions(self.estimator_transition)
        self.estimator_transition.clear()

    def update_estimator(self):
        if self.estimator_storage.step == 0:
            return {}

        metrics = {}
        for batch in self.estimator_storage.mini_batch_generator(
            self._num_mini_batches_jepa, self._num_learning_epochs_jepa
        ):
            loss_dict = self._estimator.update(batch)
            for k, v in loss_dict.items():
                metrics.setdefault(k, []).append(v)

        self.estimator_storage.clear()
        return {f"jepa_{k}": sum(v) / len(v) for k, v in metrics.items()}

    def _save_checkpoint_and_export(self, filename: str, obs, depth_stack):
        """Save a checkpoint and matching ONNX models and samples on the logging process."""
        export_error = None
        if self.log_dir is not None and not self.disable_logs:
            model_dir = os.path.join(self.log_dir, "exported")

            try:
                self.save(os.path.join(self.log_dir, filename))
                self._clear_previous_onnx_export(model_dir)
                self.export_policy_onnx(model_dir)
                self._export_onnx_test_cases(model_dir, obs, depth_stack)
            except Exception as exc:
                export_error = exc
            finally:
                # ONNX export switches the live models to eval mode.
                self.train_mode()

        if self.is_distributed:
            export_failed = torch.tensor(
                int(export_error is not None), dtype=torch.int32, device=self.device
            )
            torch.distributed.broadcast(export_failed, src=0)
            if export_failed.item() and export_error is None:
                export_error = RuntimeError("Checkpoint save or ONNX export failed on rank 0.")

        if export_error is not None:
            raise export_error

    @staticmethod
    def _clear_previous_onnx_export(model_dir: str):
        for filename in ("policy.onnx", "sensor_estimator.onnx"):
            model_path = os.path.join(model_dir, filename)
            if os.path.isfile(model_path) or os.path.islink(model_path):
                os.remove(model_path)

        test_case_dir = os.path.join(model_dir, "test_cases")
        if os.path.islink(test_case_dir):
            os.remove(test_case_dir)
        elif os.path.isdir(test_case_dir):
            shutil.rmtree(test_case_dir)

    def _export_onnx_test_cases(self, model_dir: str, obs, depth_stack):
        sample_count = min(self.onnx_test_samples, obs.shape[0])
        if sample_count < 1:
            raise RuntimeError("Cannot export ONNX test cases from an empty observation batch.")

        inference_policy = self.get_inference_policy(
            onnx_test_case_dir=os.path.join(model_dir, "test_cases"),
            onnx_test_case_samples=sample_count,
            onnx_test_case_sample_probability=1.0,
        )
        inference_policy(obs, {"observations": {"depth": depth_stack}})
        if not inference_policy.onnx_test_cases_complete:
            raise RuntimeError("Failed to export the requested ONNX test cases.")

    def save(self, path: str, infos=None):
        env = getattr(self.env, "unwrapped", self.env)
        if hasattr(env, "get_curriculum_state"):
            curriculum_state = env.get_curriculum_state()
        else:
            curriculum_state = {"common_step_counter": int(env.common_step_counter)}

        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "student_encoder_optimizer_state_dict": self.alg.optimizer_student_encoder.state_dict(),
            "estimator_state_dict": self._estimator.state_dict(),
            "estimator_optimizer_state_dict": self._estimator.optimizer.state_dict(),
            # ``iter`` remains the zero-based, last completed iteration for
            # compatibility with existing tooling and checkpoint filenames.
            "iter": self.current_learning_iteration - 1,
            "completed_iterations": self.current_learning_iteration,
            "curriculum_state": curriculum_state,
            "tot_timesteps": self.tot_timesteps,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        torch.save(saved_dict, path)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        state_dict = loaded_dict["model_state_dict"]
        is_cts_checkpoint = "student_encoder_optimizer_state_dict" in loaded_dict or any(
            key.startswith("student_encoder.") for key in state_dict
        )

        if is_cts_checkpoint:
            self.alg.policy.load_state_dict(state_dict)
        else:
            missing, unexpected = torch.nn.Module.load_state_dict(self.alg.policy, state_dict, strict=False)
            bad_missing = [key for key in missing if not key.startswith("student_encoder.")]
            if bad_missing or unexpected:
                raise RuntimeError(
                    "Teacher checkpoint does not match the CTS teacher modules. "
                    f"Missing: {bad_missing}. Unexpected: {unexpected}."
                )
            print(
                "[INFO] Loaded teacher-only checkpoint into CTS policy; student encoder remains randomly initialized."
            )

        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])

        if load_optimizer and is_cts_checkpoint:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.optimizer_student_encoder.load_state_dict(loaded_dict["student_encoder_optimizer_state_dict"])
            # Adam restores the param-group learning rate, while the adaptive
            # KL schedule also keeps a scalar copy on the algorithm.
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]

        if "estimator_state_dict" in loaded_dict:
            self._estimator.load_state_dict(loaded_dict["estimator_state_dict"])
            if load_optimizer and is_cts_checkpoint and "estimator_optimizer_state_dict" in loaded_dict:
                self._estimator.optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])

        if "completed_iterations" in loaded_dict:
            self.current_learning_iteration = int(loaded_dict["completed_iterations"])
        elif "iter" in loaded_dict:
            # Legacy checkpoints stored the last completed zero-based
            # iteration.  The next rollout is therefore ``iter + 1``.
            self.current_learning_iteration = int(loaded_dict["iter"]) + 1
        else:
            self.current_learning_iteration = 0

        self.alg.set_iteration(self.current_learning_iteration)

        if load_optimizer and is_cts_checkpoint:
            curriculum_state = loaded_dict.get("curriculum_state")
            if curriculum_state is None:
                # Older CTS checkpoints did not persist environment state.
                # Every training iteration contains exactly
                # ``num_steps_per_env`` environment steps, so this recovers all
                # iteration-based curricula and periodic sensor schedules.
                print(
                    "[WARNING] This legacy checkpoint has no saved adaptive terrain levels. "
                    "Global iteration-based curricula will be restored, but terrain levels "
                    "will restart from their initial values."
                )
                curriculum_state = {
                    "common_step_counter": self.current_learning_iteration * self.num_steps_per_env
                }

            env = getattr(self.env, "unwrapped", self.env)
            if hasattr(env, "load_curriculum_state"):
                env.load_curriculum_state(curriculum_state)
            else:
                env.common_step_counter = int(curriculum_state["common_step_counter"])

            default_timesteps = (
                self.current_learning_iteration
                * self.num_steps_per_env
                * self.env.num_envs
                * self.gpu_world_size
            )
            self.tot_timesteps = int(loaded_dict.get("tot_timesteps", default_timesteps))
            print(
                "[INFO] Resuming CTS training at iteration "
                f"{self.current_learning_iteration} with environment step counter "
                f"{env.common_step_counter}."
            )

        return loaded_dict.get("infos")

    def train_mode(self):
        self.alg.policy.train()
        self._estimator.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        self.alg.policy.eval()
        self._estimator.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def get_teacher_inference_policy(self, device=None):
        return super().get_inference_policy(device=device)

    def get_inference_policy(
        self,
        device=None,
        onnx_test_case_dir: str | None = None,
        onnx_test_case_samples: int = 0,
        onnx_test_case_seed: int | None = None,
        onnx_test_case_sample_probability: float = 0.2,
        onnx_test_case_log_interval: int = 25,
    ):
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
            self._estimator.to(device)
            if self.empirical_normalization:
                self.obs_normalizer.to(device)

        class _OnnxTestCaseRecorder:
            def __init__(self, root_dir, max_samples, seed=None, sample_probability=0.2, log_interval=25):
                self.root_dir = root_dir
                self.max_samples = max_samples
                self.sample_probability = sample_probability
                self.log_interval = max(1, log_interval)
                self.rng = np.random.default_rng(seed)
                self.counts = {"policy": 0, "sensor_estimator": 0}
                self.global_step = 0
                os.makedirs(os.path.join(root_dir, "policy"), exist_ok=True)
                os.makedirs(os.path.join(root_dir, "sensor_estimator"), exist_ok=True)
                self._clear_previous_samples()

            def _clear_previous_samples(self):
                for model_name in self.counts:
                    sample_dir = os.path.join(self.root_dir, model_name)
                    for filename in os.listdir(sample_dir):
                        if filename.startswith("sample_") and filename.endswith(".npz"):
                            os.remove(os.path.join(sample_dir, filename))

            @property
            def complete(self):
                return all(count >= self.max_samples for count in self.counts.values())

            def _eligible_env_ids(self, batch_size, model_name):
                remaining = self.max_samples - self.counts[model_name]
                if remaining <= 0:
                    return []
                sampled = np.nonzero(self.rng.random(batch_size) < self.sample_probability)[0]
                if sampled.size == 0:
                    return []
                env_ids = self.rng.permutation(sampled).tolist()
                return env_ids[:remaining]

            def write_metadata(self, runner):
                policy = runner.alg.policy
                metadata = {
                    "format": "npz-per-sample",
                    "max_samples_per_model": self.max_samples,
                    "sample_probability_per_env_step": self.sample_probability,
                    "policy": {
                        "onnx_file": "../policy.onnx",
                        "sample_dir": "policy",
                        "inputs": ["proprio_hist", "high_feat"],
                        "outputs": ["actions"],
                    },
                    "sensor_estimator": {
                        "onnx_file": "../sensor_estimator.onnx",
                        "sample_dir": "sensor_estimator",
                        "inputs": ["prop_hist", "depth_stack", "hidden_in"],
                        "outputs": ["feature", "hidden_out", "depth_latent", "prop_hist_latent"],
                    },
                    "notes": [
                        "Each array includes a batch dimension of 1.",
                        "Cases are sampled from random vectorized environments while playing.",
                    ],
                    "dimensions": {
                        "num_actions": runner.env.num_actions,
                        "policy_proprio_dim": policy.student_proprio_dim,
                        "policy_proprio_hist_dim": policy.student_proprio_hist_dim,
                        "policy_high_feature_dim": policy.jepa_feature_dim,
                        "policy_high_feature_latent_dim": policy.student_encoder.HIGH_FEATURE_LATENT_DIM,
                        "estimator_prop_hist_dim": runner._estimator.prop_hist_dim,
                        "estimator_depth_shape": list(runner._estimator.get_depth_shape()),
                        "estimator_hidden_dim": runner._estimator.get_hidden_dim(),
                    },
                }
                path = os.path.join(self.root_dir, "metadata.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
                print(f"[ONNX samples] metadata written to {path}")

            def _to_numpy(self, tensor):
                return tensor.detach().cpu().numpy()

            def _write_case(self, model_name, arrays, env_id):
                sample_id = self.counts[model_name]
                path = os.path.join(self.root_dir, model_name, f"sample_{sample_id:06d}.npz")
                payload = {name: self._to_numpy(value) for name, value in arrays.items()}
                payload["env_id"] = np.array(env_id, dtype=np.int64)
                payload["play_step"] = np.array(self.global_step, dtype=np.int64)
                np.savez_compressed(path, **payload)
                self.counts[model_name] += 1
                count = self.counts[model_name]
                if count == 1 or count == self.max_samples or count % self.log_interval == 0:
                    print(f"[ONNX samples] {model_name}: wrote {count}/{self.max_samples} cases")

            def maybe_record_estimator(
                self,
                prop_hist,
                depth_stack,
                hidden_in,
                feature,
                hidden_out,
                depth_latent,
                prop_hist_latent,
            ):
                for env_id in self._eligible_env_ids(prop_hist.shape[0], "sensor_estimator"):
                    idx = slice(env_id, env_id + 1)
                    self._write_case(
                        "sensor_estimator",
                        {
                            "prop_hist": prop_hist[idx],
                            "depth_stack": depth_stack[idx],
                            "hidden_in": hidden_in[idx],
                            "feature": feature[idx],
                            "hidden_out": hidden_out[idx],
                            "depth_latent": depth_latent[idx],
                            "prop_hist_latent": prop_hist_latent[idx],
                        },
                        env_id,
                    )

            def maybe_record_policy(
                self,
                proprio_hist,
                high_feat,
                actions,
            ):
                batch_size = proprio_hist.shape[0]
                for env_id in self._eligible_env_ids(batch_size, "policy"):
                    batch_idx = slice(env_id, env_id + 1)
                    self._write_case(
                        "policy",
                        {
                            "proprio_hist": proprio_hist[batch_idx],
                            "high_feat": high_feat[batch_idx],
                            "actions": actions[batch_idx],
                        },
                        env_id,
                    )

        recorder = None
        if onnx_test_case_samples > 0:
            if onnx_test_case_dir is None:
                raise ValueError("onnx_test_case_dir must be set when onnx_test_case_samples > 0.")
            if not 0.0 < onnx_test_case_sample_probability <= 1.0:
                raise ValueError("onnx_test_case_sample_probability must be in the interval (0, 1].")
            recorder = _OnnxTestCaseRecorder(
                onnx_test_case_dir,
                max_samples=onnx_test_case_samples,
                seed=onnx_test_case_seed,
                sample_probability=onnx_test_case_sample_probability,
                log_interval=onnx_test_case_log_interval,
            )
            recorder.write_metadata(self)

        class _InferencePolicy:
            def __init__(self, runner: OnPolicyRunnerTeacherCTS, recorder=None):
                self.policy = runner.alg.policy
                self.estimator_high = runner._estimator
                self.obs_normalizer = runner.obs_normalizer
                self.empirical_normalization = runner.empirical_normalization
                self.update_interval = runner._estimator_update_interval
                self.sensor_latent_dim = runner._sensor_latent_dim
                self.step_counter = 0
                self.hidden = None
                self.sensor_latent = None
                self.recorder = recorder

            @property
            def onnx_test_cases_complete(self):
                return self.recorder is not None and self.recorder.complete

            def _run_sensor_estimator(self, prop_hist, depth_stack):
                hidden_in = self.hidden.detach().clone()
                prop_hist_latent = self.estimator_high.prop_hist_encoder(prop_hist)
                depth_padded = self.estimator_high.pad_depth(depth_stack)
                depth_latent = self.estimator_high.depth_encoder(depth_padded)
                gru_input = torch.cat([prop_hist_latent, depth_latent], dim=-1)
                hidden_out = self.estimator_high.pred_gru(gru_input, self.hidden)
                feature = hidden_out.detach()
                hidden_out = hidden_out.detach()

                if self.recorder is not None:
                    self.recorder.maybe_record_estimator(
                        prop_hist=prop_hist,
                        depth_stack=depth_stack,
                        hidden_in=hidden_in,
                        feature=feature,
                        hidden_out=hidden_out,
                        depth_latent=depth_latent,
                        prop_hist_latent=prop_hist_latent,
                    )
                return feature, hidden_out

            @torch.inference_mode()
            def __call__(self, obs, extras=None):
                B = obs.shape[0]
                dev = obs.device

                if self.hidden is None:
                    hidden_dim = self.estimator_high.get_hidden_dim()
                    self.hidden = torch.zeros(B, hidden_dim, device=dev)
                    self.sensor_latent = torch.zeros(B, self.sensor_latent_dim, device=dev)

                prop_hist = self.policy.extract_student_proprio_hist(obs)

                if extras is None:
                    raise ValueError("Inference policy requires `extras` containing depth observations.")
                depth_stack = extras["observations"]["depth"].to(dev)

                if self.step_counter % self.update_interval == 0:
                    feature_high, self.hidden = self._run_sensor_estimator(
                        prop_hist=prop_hist, depth_stack=depth_stack
                    )
                    self.sensor_latent = feature_high

                student_latent = self.policy.student_encoder(prop_hist, self.sensor_latent)
                current_proprio = prop_hist[..., : self.policy.student_proprio_dim]
                actions = self.policy.actor(torch.cat([current_proprio, student_latent], dim=-1))
                if self.recorder is not None:
                    self.recorder.maybe_record_policy(
                        proprio_hist=prop_hist,
                        high_feat=self.sensor_latent,
                        actions=actions,
                    )
                    self.recorder.global_step += 1
                self.step_counter += 1
                return actions

            def reset(self, dones):
                if self.hidden is not None:
                    self.hidden[dones] = 0
                    self.sensor_latent[dones] = 0

        return _InferencePolicy(self, recorder=recorder)

    def export_policy_onnx(self, export_dir: str):
        import copy

        import torch.nn.functional as F
        import torch.onnx

        os.makedirs(export_dir, exist_ok=True)
        self.eval_mode()

        policy = self.alg.policy
        estimator = self._estimator

        class _PolicyONNX(torch.nn.Module):
            def __init__(self, policy: ActorCriticTeacherCTS):
                super().__init__()
                self.student_encoder = copy.deepcopy(policy.student_encoder)
                self.actor = copy.deepcopy(policy.actor)
                self.student_proprio_dim = policy.student_proprio_dim
                self.student_proprio_hist_dim = policy.student_proprio_hist_dim
                self.jepa_feature_dim = policy.jepa_feature_dim
                self.num_actions = policy.num_actions

            def forward(self, proprio_hist, high_feat):
                student_latent = self.student_encoder(proprio_hist, high_feat)
                current_proprio = proprio_hist[..., : self.student_proprio_dim]
                actor_obs = torch.cat([current_proprio, student_latent], dim=-1)
                return self.actor(actor_obs)

        policy_exporter = _PolicyONNX(policy).to("cpu").eval()

        dummy_proprio_hist = torch.zeros(1, policy.student_proprio_hist_dim)
        dummy_high_feat = torch.zeros(1, policy.jepa_feature_dim)

        policy_path = os.path.join(export_dir, "policy.onnx")
        torch.onnx.export(
            policy_exporter,
            (dummy_proprio_hist, dummy_high_feat),
            policy_path,
            export_params=True,
            opset_version=20,
            input_names=["proprio_hist", "high_feat"],
            output_names=["actions"],
            dynamic_axes={},
        )

        print(f"\nPolicy exported to {policy_path}")
        print(
            f"  Inputs:  proprio_hist ({policy.student_proprio_hist_dim}), "
            f"high_feat ({policy.jepa_feature_dim})"
        )
        print(f"  Outputs: actions ({self.env.num_actions})")

        class _SensorEstimatorONNX(torch.nn.Module):
            def __init__(self, estimator: JepaEstimatorSensorSS):
                super().__init__()
                self.depth_encoder = copy.deepcopy(estimator.depth_encoder)
                self.prop_hist_encoder = copy.deepcopy(estimator.prop_hist_encoder)
                self.pred_gru = copy.deepcopy(estimator.pred_gru)
                self.raw_depth_shape = estimator.raw_depth_shape
                self.depth_shape = estimator.depth_shape

            def _pad_depth(self, depth):
                pad_h = self.depth_shape[0] - self.raw_depth_shape[0]
                pad_w = self.depth_shape[1] - self.raw_depth_shape[1]
                return F.pad(depth, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

            def forward(self, prop_hist, depth_stack, hidden_in):
                prop_hist_latent = self.prop_hist_encoder(prop_hist)
                depth_padded = self._pad_depth(depth_stack)
                depth_latent = self.depth_encoder(depth_padded)
                gru_input = torch.cat([prop_hist_latent, depth_latent], dim=-1)
                hidden_out = self.pred_gru(gru_input, hidden_in)
                return hidden_out, hidden_out, depth_latent, prop_hist_latent

        est_exporter = _SensorEstimatorONNX(estimator).to("cpu").eval()

        C_est = estimator.num_depth_channels
        H_est, W_est = estimator.raw_depth_shape
        hidden_dim = estimator.get_hidden_dim()
        dummy_prop_hist = torch.zeros(1, estimator.prop_hist_dim)
        dummy_est_depth = torch.zeros(1, C_est, H_est, W_est)
        dummy_hidden = torch.zeros(1, hidden_dim)

        est_path = os.path.join(export_dir, "sensor_estimator.onnx")
        torch.onnx.export(
            est_exporter,
            (dummy_prop_hist, dummy_est_depth, dummy_hidden),
            est_path,
            export_params=True,
            opset_version=20,
            input_names=["prop_hist", "depth_stack", "hidden_in"],
            output_names=["feature", "hidden_out", "depth_latent", "prop_hist_latent"],
            dynamic_axes={},
        )

        print(f"\nSensor estimator exported to {est_path}")
        print(
            f"  Inputs:  prop_hist ({estimator.prop_hist_dim}), "
            f"depth_stack ({C_est}, {H_est}, {W_est}), hidden_in ({hidden_dim})"
        )
        print(
            f"  Outputs: feature ({estimator.get_feature_dim()}), "
            f"hidden_out ({hidden_dim}), depth_latent ({estimator.depth_latent_dim}), "
            f"prop_hist_latent ({estimator.prop_hist_latent_dim})"
        )

    @staticmethod
    def _get_teacher_num_envs(num_envs, student_env_ratio):
        student_num_envs = int(round(num_envs * student_env_ratio))
        student_num_envs = max(0, min(num_envs - 1, student_num_envs))
        return num_envs - student_num_envs
