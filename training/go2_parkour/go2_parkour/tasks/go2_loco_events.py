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

"""Reset events that adapt RobotLab's Go2 randomization to the parkour command layout."""

from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch


def randomize_motor_zero_offset(
    env,
    env_ids: torch.Tensor | None,
    offset_range: tuple[float, float],
) -> None:
    """Apply RobotLab's per-joint motor zero-offset distribution."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env._motor_zero_offset[env_ids] = math_utils.sample_uniform(
        offset_range[0],
        offset_range[1],
        (len(env_ids), env._num_actions),
        env.device,
    )


def reset_root_state_uniform(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
) -> None:
    """Apply RobotLab's root reset distribution around this task's spawn anchors."""

    root_states = env._robot.data.default_root_state[env_ids].clone()

    # Preserve the existing terrain/command spawn semantics: waypoint terrains
    # start at their first waypoint, while direct-command terrains start at the
    # subterrain origin.
    root_states[:, :2] = env._env_goals[env_ids, 0, :2]
    root_states[:, 2] = env._env_goals[env_ids, 0, 2] + 0.4
    direct_resets = env._direct_command_terrain_mask[env_ids]
    direct_env_ids = env_ids[direct_resets]
    root_states[direct_resets, :2] = env._terrain.env_origins[direct_env_ids, :2]
    root_states[direct_resets, 2] = env._terrain.env_origins[direct_env_ids, 2] + 0.4

    pose_ranges = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=env.device,
    )
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0],
        pose_ranges[:, 1],
        (len(env_ids), 6),
        env.device,
    )
    waypoint_resets = env._waypoint_command_terrain_mask[env_ids]
    pose_samples[waypoint_resets, :3] = 0.0
    pose_samples[waypoint_resets, 5] = 0.0
    positions = root_states[:, :3] + pose_samples[:, :3]
    orientation_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3],
        pose_samples[:, 4],
        pose_samples[:, 5],
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientation_delta)

    velocity_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=env.device,
    )
    velocity_samples = math_utils.sample_uniform(
        velocity_ranges[:, 0],
        velocity_ranges[:, 1],
        (len(env_ids), 6),
        env.device,
    )
    velocities = root_states[:, 7:13] + velocity_samples

    env._robot.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    env._robot.write_root_velocity_to_sim(velocities, env_ids=env_ids)
