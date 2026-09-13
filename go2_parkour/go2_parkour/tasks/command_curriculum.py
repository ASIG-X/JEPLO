from __future__ import annotations

import torch


def update_direct_command_max_distance(
    max_move_distance: torch.Tensor,
    robot_positions_xy: torch.Tensor,
    spawn_positions_xy: torch.Tensor,
    direct_command_mask: torch.Tensor,
) -> None:
    """Update each direct-command episode's maximum planar distance from its spawn point."""
    current_distance = torch.linalg.norm(robot_positions_xy - spawn_positions_xy, dim=1)
    max_move_distance[direct_command_mask] = torch.maximum(
        max_move_distance[direct_command_mask],
        current_distance[direct_command_mask],
    )


def direct_command_curriculum_moves(
    max_move_distance: torch.Tensor,
    commands_xy_accumulation: torch.Tensor,
    terrain_length: float,
    resampling_time: float,
    command_not_zero_probability: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RobotLab-style terrain curriculum moves for direct commands."""
    target_distance = (
        torch.linalg.norm(commands_xy_accumulation, dim=1)
        * resampling_time
        * command_not_zero_probability
    )
    move_up = max_move_distance > terrain_length / 2.0
    move_down = (max_move_distance < target_distance * 0.5) & ~move_up
    return move_up, move_down
