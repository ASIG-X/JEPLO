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

"""RobotLab Go2 reward formulas adapted to the direct parkour environment."""

from __future__ import annotations

import torch


def linear_schedule(
    iteration: int,
    initial: float,
    final: float,
    start_iteration: int,
    end_iteration: int,
) -> float:
    """Match RobotLab's linear reward-weight curriculum."""

    if iteration < start_iteration:
        return initial
    if iteration >= end_iteration:
        return final
    progress = (iteration - start_iteration) / (end_iteration - start_iteration)
    return initial + progress * (final - initial)


def track_lin_vel_xy_exp(command: torch.Tensor, root_lin_vel_b: torch.Tensor, std: float = 0.5) -> torch.Tensor:
    """Reward exact tracking of the commanded XY velocity in the body frame."""
    error = torch.sum(torch.square(command[:, :2] - root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-error / std**2)


def track_ang_vel_z_exp(command: torch.Tensor, root_ang_vel_b: torch.Tensor, std: float = 0.5) -> torch.Tensor:
    error = torch.square(command[:, 2] - root_ang_vel_b[:, 2])
    return torch.exp(-error / std**2)


def action_smoothness_l2(
    action: torch.Tensor,
    previous_action: torch.Tensor,
    previous_previous_action: torch.Tensor,
) -> torch.Tensor:
    """Match RobotLab's element-wise masking of the first two action-history steps."""

    difference = torch.square(action - 2.0 * previous_action + previous_previous_action)
    difference *= previous_action != 0
    difference *= previous_previous_action != 0
    return torch.sum(difference, dim=1)


def base_height(
    root_pos_w: torch.Tensor,
    ray_hits_w: torch.Tensor,
    target_height: float,
) -> torch.Tensor:
    """Estimate base height using RobotLab's invalid-scan fallback."""

    ray_hits_z = ray_hits_w[..., 2]
    invalid = (
        torch.isnan(ray_hits_z).any(dim=1)
        | torch.isinf(ray_hits_z).any(dim=1)
        | (torch.max(torch.abs(ray_hits_z), dim=1).values > 1.0e6)
    )
    base_z = root_pos_w[:, 2]
    estimated_ground_z = torch.mean(ray_hits_z, dim=1)
    fallback_ground_z = base_z - target_height
    estimated_ground_z = torch.where(invalid, fallback_ground_z, estimated_ground_z)
    return base_z - estimated_ground_z


def undesired_contacts(net_contact_forces_w_history: torch.Tensor, threshold: float) -> torch.Tensor:
    contact = torch.max(torch.norm(net_contact_forces_w_history, dim=-1), dim=1).values > threshold
    return torch.sum(contact, dim=1).float()


def feet_regulation(
    feet_pos_w: torch.Tensor,
    base_pos_w: torch.Tensor,
    feet_lin_vel_w: torch.Tensor,
    measured_base_height: torch.Tensor,
    gravity: tuple[float, float, float],
    base_height_target: float,
) -> torch.Tensor:
    """Penalize fast horizontal foot motion near the ground exactly as RobotLab does."""

    gravity_w = torch.tensor(gravity, device=feet_pos_w.device, dtype=feet_pos_w.dtype)
    down_w = gravity_w / torch.norm(gravity_w)
    delta_feet_w = feet_pos_w - base_pos_w.unsqueeze(1)
    feet_to_base_height = torch.sum(delta_feet_w * down_w.view(1, 1, 3), dim=-1)
    feet_height = torch.clamp(measured_base_height.unsqueeze(1) - feet_to_base_height, min=0.0)
    horizontal_speed_sq = feet_lin_vel_w[..., :2].pow(2).sum(dim=-1)
    return (horizontal_speed_sq * torch.exp(-feet_height / (0.025 * base_height_target))).sum(dim=-1)


def hip_pos_penalty_l1(
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    command: torch.Tensor,
    stand_still_scale: float = 1.0,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    command_large = torch.any(torch.abs(command[:, [1, 2]]) > command_threshold, dim=1)
    running = torch.linalg.norm(joint_pos - default_joint_pos, dim=1, ord=1)
    return torch.where(command_large, running, stand_still_scale * running)


def joint_pos_penalty_l1(
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    command: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    stand_still_scale: float = 1.0,
    velocity_threshold: float = 0.1,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    command_norm = torch.linalg.norm(command, dim=1)
    body_speed = torch.linalg.norm(root_lin_vel_b[:, :2], dim=1)
    running = torch.linalg.norm(joint_pos - default_joint_pos, dim=1, ord=1)
    moving = torch.logical_or(command_norm > command_threshold, body_speed > velocity_threshold)
    return torch.where(moving, running, stand_still_scale * running)
