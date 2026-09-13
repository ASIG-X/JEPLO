# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast sensor."""

from __future__ import annotations

from collections.abc import Callable

import torch
from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass


def ray_spherical_slice_pattern(cfg: "RaySphericalSlicePatternCfg", device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Spherical slice ray pattern for generating a 2D heightmap-like scan.

    This pattern creates rays in a spherical slice pattern, useful for converting
    Mid360 LiDAR point clouds to dense spherical scans.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions and directions of the rays.
    """

    h_res = (cfg.horizontal_fov_deg[1] - cfg.horizontal_fov_deg[0]) / cfg.horizontal_num_rays
    v_res = (cfg.vertical_fov_deg[1] - cfg.vertical_fov_deg[0]) / cfg.vertical_num_rays

    h_low_deg = cfg.horizontal_fov_deg[0] + h_res / 2.0
    v_low_deg = cfg.vertical_fov_deg[0] + v_res / 2.0

    h_high_deg = cfg.horizontal_fov_deg[1] - h_res / 2.0
    v_high_deg = cfg.vertical_fov_deg[1] - v_res / 2.0

    # first index  -> row:    one horizontal line of rays
    # second index -> column: one vertical line of rays
    horizontal_angles_rad = [
        torch.deg2rad(torch.linspace(h_low_deg, h_high_deg, cfg.horizontal_num_rays))
        for _ in range(cfg.vertical_num_rays)
    ]
    vertical_angles_rad = [
        torch.full((cfg.horizontal_num_rays,), angle)
        for angle in torch.deg2rad(torch.linspace(v_low_deg, v_high_deg, cfg.vertical_num_rays))
    ]

    v_angles = torch.cat(vertical_angles_rad, dim=0)
    h_angles = torch.cat(horizontal_angles_rad, dim=0)

    x = torch.cos(v_angles) * torch.cos(h_angles)
    y = torch.cos(v_angles) * torch.sin(h_angles)
    z = torch.sin(v_angles)

    ray_directions = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)
    ray_starts = torch.zeros_like(ray_directions).to(device)

    return ray_starts, ray_directions


@configclass
class RaySphericalSlicePatternCfg(PatternBaseCfg):
    """Configuration for a spherical slice ray pattern.

    This pattern generates rays in a 2D grid pattern on a spherical surface,
    covering specified vertical and horizontal FOV ranges. The default values
    are configured for the Mid360 LiDAR front-facing scan with 27x180 resolution.
    """

    func: Callable = ray_spherical_slice_pattern
    vertical_fov_deg: tuple[float, float] = (-4.5, 45.5)
    vertical_num_rays: int = 25
    horizontal_fov_deg: tuple[float, float] = (-60.0, 60.0)
    horizontal_num_rays: int = 60
