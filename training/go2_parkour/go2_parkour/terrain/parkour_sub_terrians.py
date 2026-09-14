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

import random

import numpy as np
import scipy.interpolate as interpolate
from go2_parkour.terrain import parkour_sub_terrains_cfg as parkour_terrains_cfg
from go2_parkour.terrain.utils import parkour_field_to_mesh


def padding_height_field_raw(
    height_field_raw: np.ndarray, cfg: parkour_terrains_cfg.ExtremeParkourRoughTerrainCfg
) -> np.ndarray:
    pad_width = int(cfg.pad_width // cfg.horizontal_scale)
    pad_height = int(cfg.pad_height // cfg.vertical_scale)
    height_field_raw[:, :pad_width] = pad_height
    height_field_raw[:, -pad_width:] = pad_height
    height_field_raw[:pad_width, :] = pad_height
    height_field_raw[-pad_width:, :] = pad_height
    height_field_raw = np.rint(height_field_raw).astype(np.int16)
    return height_field_raw


def random_uniform_terrain(
    difficulty: float,
    cfg: parkour_terrains_cfg.ExtremeParkourRoughTerrainCfg,
    height_field_raw: np.ndarray,
):
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    max_height = (cfg.noise_range[1] - cfg.noise_range[0]) * difficulty + cfg.noise_range[0]
    height_min = int(-cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(max_height / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)
    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    z_upsampled = np.rint(z_upsampled).astype(np.int16)
    height_field_raw += z_upsampled
    return height_field_raw


def clear_height_field_rect(
    height_field_raw: np.ndarray,
    center_x: int,
    center_y: int,
    half_size_x: int,
    half_size_y: int,
    height: int | bool = 0,
) -> None:
    width_pixels, length_pixels = height_field_raw.shape
    x_lo = max(0, center_x - half_size_x)
    x_hi = min(width_pixels, center_x + half_size_x)
    y_lo = max(0, center_y - half_size_y)
    y_hi = min(length_pixels, center_y + half_size_y)
    if x_hi > x_lo and y_hi > y_lo:
        height_field_raw[x_lo:x_hi, y_lo:y_hi] = height


def fill_rotated_height_field_square(
    height_field_raw: np.ndarray,
    center_x: float,
    center_y: float,
    size: float,
    angle: float,
    height: int,
) -> None:
    """Fill a square rotated around its center in height-field pixel coordinates."""
    width_pixels, length_pixels = height_field_raw.shape
    half_size = size / 2.0
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    bounding_half_size = half_size * (abs(cos_angle) + abs(sin_angle))

    x_lo = max(0, int(np.floor(center_x - bounding_half_size)))
    x_hi = min(width_pixels, int(np.ceil(center_x + bounding_half_size)) + 1)
    y_lo = max(0, int(np.floor(center_y - bounding_half_size)))
    y_hi = min(length_pixels, int(np.ceil(center_y + bounding_half_size)) + 1)
    if x_hi <= x_lo or y_hi <= y_lo:
        return

    x_offset = np.arange(x_lo, x_hi)[:, None] - center_x
    y_offset = np.arange(y_lo, y_hi)[None, :] - center_y
    local_x = cos_angle * x_offset + sin_angle * y_offset
    local_y = -sin_angle * x_offset + cos_angle * y_offset
    square_mask = (np.abs(local_x) <= half_size) & (np.abs(local_y) <= half_size)
    height_field_raw[x_lo:x_hi, y_lo:y_hi][square_mask] = height


def compute_centerline_goals(
    height_field_raw: np.ndarray,
    cfg: parkour_terrains_cfg.ExtremeParkourRoughTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width_pixels, length_pixels = height_field_raw.shape
    mid_y = length_pixels // 2
    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    goal_start_x = platform_len - round(1.0 / cfg.horizontal_scale)
    goal_end_x = width_pixels - round(0.5 / cfg.horizontal_scale)
    goal_x_positions = np.linspace(goal_start_x, goal_end_x, num_goals)
    zigzag_y_range = getattr(cfg, "zigzag_y_range", None)
    zigzag_offset = 0
    y_min = 0
    y_max = length_pixels - 1
    if zigzag_y_range is not None:
        zigzag_offset = round(np.random.uniform(zigzag_y_range[0], zigzag_y_range[1]) / cfg.horizontal_scale)
        zigzag_y_margin = getattr(cfg, "zigzag_y_margin", 0.3)
        y_margin = round(zigzag_y_margin / cfg.horizontal_scale)
        y_min = min(max(0, y_margin), max(0, length_pixels - 1))
        y_max = max(y_min, length_pixels - y_margin - 1)

    goals = np.zeros((num_goals, 2))
    goal_heights = np.zeros(num_goals)
    goal_y_shift_ranges = np.zeros(num_goals)
    for goal_idx, goal_x in enumerate(goal_x_positions):
        goal_px = int(np.clip(round(goal_x), 0, width_pixels - 1))
        goal_y = mid_y
        if zigzag_y_range is not None and 0 < goal_idx < num_goals - 1:
            direction = -1 if goal_idx % 2 else 1
            goal_y = mid_y + direction * zigzag_offset
        goal_py = int(np.clip(goal_y, y_min, y_max))
        goals[goal_idx] = [goal_px, goal_py]
        goal_heights[goal_idx] = height_field_raw[goal_px, goal_py]

    return goals, goal_heights, goal_y_shift_ranges


@parkour_field_to_mesh
def parkour_gap_terrain(
    difficulty: float,
    cfg: parkour_terrains_cfg.ExtremeParkourGapTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))
    mid_y = length_pixels // 2  # length is actually y width
    gap_size = eval(cfg.gap_size, {"difficulty": difficulty})
    gap_size = max(1, round(gap_size / cfg.horizontal_scale))
    if cfg.max_num_gaps < 1:
        raise ValueError(f"max_num_gaps must be positive, got {cfg.max_num_gaps}.")

    spacing_min = round(cfg.x_range[0] / cfg.horizontal_scale)
    spacing_max = round(cfg.x_range[1] / cfg.horizontal_scale)
    if spacing_min < 1 or spacing_max < spacing_min:
        raise ValueError(f"Invalid gap spacing range: {cfg.x_range}.")

    dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
    dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height

    gap_depth_progress = difficulty**cfg.gap_depth_difficulty_power
    gap_depth_min = cfg.gap_depth_start[0] + (cfg.gap_depth[0] - cfg.gap_depth_start[0]) * gap_depth_progress
    gap_depth_max = cfg.gap_depth_start[1] + (cfg.gap_depth[1] - cfg.gap_depth_start[1]) * gap_depth_progress
    gap_depth = -round(np.random.uniform(gap_depth_min, gap_depth_max) / cfg.vertical_scale)
    half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)
    goals = np.zeros((num_goals, 2))
    goal_heights = np.ones((num_goals)) * platform_height
    x_edge_mask = np.zeros((width_pixels, length_pixels), dtype=bool)

    # Use the largest number of gaps that fits after the randomized starting
    # platform while preserving the configured ending platform.
    start_platform_len = round(np.random.uniform(1.0, 2.0) / cfg.horizontal_scale)
    final_platform_start = width_pixels - platform_len
    gap_area_length = final_platform_start - platform_len
    edge_clearance = max(1, spacing_min // 2)
    num_gaps = min(cfg.max_num_gaps, num_goals - 2)
    while num_gaps > 0:
        minimum_length = (
            start_platform_len + num_gaps * gap_size + (num_gaps - 1) * spacing_min + edge_clearance
        )
        if minimum_length <= gap_area_length:
            break
        num_gaps -= 1
    if num_gaps == 0:
        raise ValueError("The gap area is too short for the configured gap geometry.")

    # Randomize internal clear distances and use all remaining space for the
    # trailing corridor, so the obstacle area fills its longitudinal span.
    extra_budget = (
        gap_area_length
        - start_platform_len
        - num_gaps * gap_size
        - (num_gaps - 1) * spacing_min
        - edge_clearance
    )
    internal_spacings = []
    for _ in range(num_gaps - 1):
        max_extra = min(spacing_max - spacing_min, extra_budget)
        extra = np.random.randint(0, max_extra + 1) if max_extra > 0 else 0
        internal_spacings.append(spacing_min + extra)
        extra_budget -= extra

    trailing_corridor_len = gap_area_length - start_platform_len - num_gaps * gap_size - sum(internal_spacings)
    corridor_lengths = [start_platform_len, *internal_spacings, trailing_corridor_len]
    corridor_y_offsets = np.random.randint(dis_y_min, dis_y_max, size=num_gaps + 1)
    goals[0] = [platform_len, mid_y + corridor_y_offsets[0]]

    dis_x = platform_len
    for i in range(num_gaps):
        corridor_start = dis_x
        corridor_end = corridor_start + corridor_lengths[i]
        rand_y = corridor_y_offsets[i]
        corridor_y_lo = mid_y + rand_y - half_valid_width
        corridor_y_hi = mid_y + rand_y + half_valid_width
        height_field_raw[corridor_start:corridor_end, :corridor_y_lo] = gap_depth
        height_field_raw[corridor_start:corridor_end, corridor_y_hi:] = gap_depth
        goal_x = corridor_end - 0.5 / cfg.horizontal_scale if i == 0 else (corridor_start + corridor_end) // 2
        goals[i + 1] = [goal_x, mid_y + rand_y]

        gap_start = corridor_end
        gap_end = gap_start + gap_size
        if not cfg.apply_flat:
            height_field_raw[gap_start:gap_end, :] = gap_depth
            next_corridor_y_lo = mid_y + corridor_y_offsets[i + 1] - half_valid_width
            next_corridor_y_hi = mid_y + corridor_y_offsets[i + 1] + half_valid_width
            x_edge_mask[gap_start:gap_end, :] = True
            x_edge_mask[gap_start - 1 : gap_start, corridor_y_lo:corridor_y_hi] = True
            x_edge_mask[gap_end : gap_end + 1, next_corridor_y_lo:next_corridor_y_hi] = True
        dis_x = gap_end

    corridor_end = dis_x + corridor_lengths[-1]
    rand_y = corridor_y_offsets[-1]
    corridor_y_lo = mid_y + rand_y - half_valid_width
    corridor_y_hi = mid_y + rand_y + half_valid_width
    height_field_raw[dis_x:corridor_end, :corridor_y_lo] = gap_depth
    height_field_raw[dis_x:corridor_end, corridor_y_hi:] = gap_depth
    if corridor_end != final_platform_start:
        raise RuntimeError("Gap layout does not fill the configured obstacle area.")

    height_field_raw[final_platform_start:, :] = platform_height
    last_gap_start = dis_x - gap_size
    final_goal_x = dis_x + last_gap_start - goals[num_gaps, 0]
    goals[num_gaps + 1] = [final_goal_x, mid_y + rand_y]
    goals[num_gaps + 2 :] = [width_pixels - 1, mid_y]
    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
    goal_y_shift_ranges = np.zeros(num_goals)
    return (
        height_field_raw,
        goals * cfg.horizontal_scale,
        goal_heights * cfg.vertical_scale,
        goal_y_shift_ranges,
        x_edge_mask,
    )


@parkour_field_to_mesh
def parkour_hurdle_terrain(
    difficulty: float,
    cfg: parkour_terrains_cfg.ExtremeParkourHurdleTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stone_len = eval(cfg.stone_len, {"difficulty": difficulty})
    stone_len = round(stone_len / cfg.horizontal_scale)

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))

    mid_y = length_pixels // 2  # length is actually y width
    dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
    dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale)
    dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
    dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

    half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)
    hurdle_height_range = eval(cfg.hurdle_height_range, {"difficulty": difficulty})
    hurdle_height_max = round(hurdle_height_range[1] / cfg.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / cfg.vertical_scale)

    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height
    dis_x = platform_len
    goals = np.zeros((num_goals, 2))
    goal_heights = np.ones((num_goals)) * platform_height

    goals[0] = [platform_len - 1, mid_y]

    for i in range(num_goals - 2):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        if not cfg.apply_flat:
            height_field_raw[dis_x - stone_len // 2 : dis_x + stone_len // 2,] = np.random.randint(
                hurdle_height_min, hurdle_height_max
            )
            height_field_raw[dis_x - stone_len // 2 : dis_x + stone_len // 2, : mid_y + rand_y - half_valid_width] = 0
            height_field_raw[dis_x - stone_len // 2 : dis_x + stone_len // 2, mid_y + rand_y + half_valid_width :] = 0
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)

    if final_dis_x > width_pixels:
        final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
    goal_y_shift_ranges = np.zeros(num_goals)
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale, goal_y_shift_ranges, None


@parkour_field_to_mesh
def parkour_step_terrain(
    difficulty: float,
    cfg: parkour_terrains_cfg.ExtremeParkourStepTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step_height = eval(cfg.step_height, {"difficulty": difficulty})
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))

    mid_y = length_pixels // 2  # length is actually y width
    dis_x_min = round(cfg.x_range[0] / cfg.horizontal_scale)
    dis_x_max = round(cfg.x_range[1] / cfg.horizontal_scale)
    dis_y_min = round(cfg.y_range[0] / cfg.horizontal_scale)
    dis_y_max = round(cfg.y_range[1] / cfg.horizontal_scale)

    step_height = round(step_height / cfg.vertical_scale)

    half_valid_width = round(np.random.uniform(cfg.half_valid_width[0], cfg.half_valid_width[1]) / cfg.horizontal_scale)

    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height

    dis_x = platform_len
    last_dis_x = dis_x
    stair_height = 0
    goals = np.zeros((num_goals, 2))
    goals[0] = [platform_len - round(1 / cfg.horizontal_scale), mid_y]
    goal_heights = np.ones((num_goals)) * platform_height

    num_stones = num_goals - 2
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        rand_y = np.random.randint(dis_y_min, dis_y_max)
        if i < num_stones // 2:
            stair_height += step_height
        elif i > num_stones // 2:
            stair_height -= step_height
        height_field_raw[dis_x : dis_x + rand_x,] = stair_height
        dis_x += rand_x
        height_field_raw[last_dis_x:dis_x, : mid_y + rand_y - half_valid_width] = 0
        height_field_raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width :] = 0

        last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
        goal_heights[i + 1] = stair_height
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > width_pixels:
        final_dis_x = width_pixels - 0.5 // cfg.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]
    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
    goal_y_shift_ranges = np.zeros(num_goals)
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights, goal_y_shift_ranges, None


@parkour_field_to_mesh
def parkour_terrain(
    difficulty: float,
    cfg: parkour_terrains_cfg.ExtremeParkourTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))
    height_field_raw[:] = -round(np.random.uniform(cfg.pit_depth[0], cfg.pit_depth[1]) / cfg.vertical_scale)
    mid_y = length_pixels // 2  # length is actually y width
    stone_len = eval(cfg.stone_len, {"difficulty": difficulty})
    stone_len = np.random.uniform(*stone_len)
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len = round(stone_len / cfg.horizontal_scale)
    x_range = eval(cfg.x_range, {"difficulty": difficulty})
    y_range = eval(cfg.y_range, {"difficulty": difficulty})
    dis_x_min = stone_len + round(x_range[0] / cfg.horizontal_scale)
    dis_x_max = stone_len + round(x_range[1] / cfg.horizontal_scale)
    dis_y_min = round(y_range[0] / cfg.horizontal_scale)
    dis_y_max = round(y_range[1] / cfg.horizontal_scale)

    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height

    stone_width = round(cfg.stone_width / cfg.horizontal_scale)
    last_stone_len = round(cfg.last_stone_len / cfg.horizontal_scale)

    incline_height = eval(cfg.incline_height, {"difficulty": difficulty})
    last_incline_height = eval(cfg.last_incline_height, {"difficulty": difficulty, "incline_height": incline_height})
    last_incline_height = round(last_incline_height / cfg.vertical_scale)
    incline_height = round(incline_height / cfg.vertical_scale)

    dis_x = platform_len - np.random.randint(dis_x_min, dis_x_max) + stone_len // 2
    goals = np.zeros((num_goals, 2))
    goal_heights = np.ones((num_goals)) * platform_height
    goals[0] = [platform_len - stone_len // 2, mid_y]
    left_right_flag = np.random.randint(0, 2)
    dis_z = 0
    num_stones = num_goals - 2
    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2 * (left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len // 4
            heights = (
                np.tile(np.linspace(-last_incline_height, last_incline_height, stone_width), (last_stone_len, 1))
                * pos_neg
            )
            height_field_raw[
                dis_x - last_stone_len // 2 : dis_x + last_stone_len // 2,
                dis_y - stone_width // 2 : dis_y + stone_width // 2,
            ] = heights.astype(int) + dis_z
        else:
            heights = np.tile(np.linspace(-incline_height, incline_height, stone_width), (stone_len, 1)) * pos_neg
            height_field_raw[
                dis_x - stone_len // 2 : dis_x + stone_len // 2, dis_y - stone_width // 2 : dis_y + stone_width // 2
            ] = heights.astype(int) + dis_z

        goals[i + 1] = [dis_x, dis_y]
        goal_heights[i + 1] = np.mean(heights.astype(int))

        left_right_flag = 1 - left_right_flag
    final_dis_x = dis_x + 2 * np.random.randint(dis_x_min, dis_x_max)
    final_platform_start = dis_x + last_stone_len // 2 + round(0.05 // cfg.horizontal_scale)
    height_field_raw[final_platform_start:, :] = platform_height
    goals[-1] = [final_dis_x, mid_y]
    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    goal_y_shift_ranges = np.zeros(num_goals)
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale, goal_y_shift_ranges, None


@parkour_field_to_mesh
def parkour_stairs_terrain(
    difficulty: float,
    cfg: parkour_terrains_cfg.ExtremeParkourStairsTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate centered sets of uniform up-down stairs with waypoints before, on, and after each set."""
    step_height = eval(cfg.step_height, {"difficulty": difficulty})
    step_height_pixels = round(step_height / cfg.vertical_scale)
    # Randomize step depth within range (fixed for this subterrain)
    step_depth = np.random.uniform(cfg.step_depth_range[0], cfg.step_depth_range[1])
    step_depth_pixels = round(step_depth / cfg.horizontal_scale)
    # Randomize stair width within range (fixed for this subterrain)
    stair_width = np.random.uniform(cfg.stair_width_range[0], cfg.stair_width_range[1])
    stair_width_pixels = round(stair_width / cfg.horizontal_scale)
    flat_spacing_pixels = round(cfg.flat_spacing / cfg.horizontal_scale)

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))

    mid_y = length_pixels // 2
    stair_y_lo = max(0, mid_y - stair_width_pixels // 2)
    stair_y_hi = min(length_pixels, mid_y + stair_width_pixels // 2)
    edge_band_pixels = max(2, int(np.ceil(step_depth_pixels * 0.25)))
    x_edge_mask = np.zeros((width_pixels, length_pixels), dtype=bool)

    def mark_edge_band(x_lo: int, x_hi: int):
        x_lo = max(0, x_lo)
        x_hi = min(width_pixels, x_hi)
        if x_hi > x_lo and stair_y_hi > stair_y_lo:
            x_edge_mask[x_lo:x_hi, stair_y_lo:stair_y_hi] = True

    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height

    top_platform_lengths = [
        round(np.random.uniform(0.3, 0.6) / cfg.horizontal_scale) for _ in range(cfg.num_stair_sets)
    ]
    stair_set_length = 2 * cfg.num_steps * step_depth_pixels
    total_stairs_length = (
        cfg.num_stair_sets * stair_set_length
        + sum(top_platform_lengths)
        + max(0, cfg.num_stair_sets - 1) * flat_spacing_pixels
    )
    dis_x = max(0, (width_pixels - total_stairs_length) // 2)
    first_stair_x = dis_x

    # Collect all waypoint positions and heights first
    waypoint_positions = []
    waypoint_heights = []

    # Create multiple sets of stairs (up-down)
    for stair_set in range(cfg.num_stair_sets):
        current_height = 0

        # Waypoint before the ascending stairs.
        pre_stair_spacing = round(0.3 / cfg.horizontal_scale)
        waypoint_positions.append([max(0, dis_x - pre_stair_spacing), mid_y])
        waypoint_heights.append(0)

        # Create stairs going up
        for i in range(cfg.num_steps):
            step_start = dis_x
            current_height += step_height_pixels
            height_field_raw[step_start : step_start + step_depth_pixels, stair_y_lo:stair_y_hi] = current_height
            mark_edge_band(step_start, step_start + edge_band_pixels)
            dis_x += step_depth_pixels

        # Flat platform at the top (randomized between 0.3m and 0.6m).
        top_platform_len = top_platform_lengths[stair_set]
        height_field_raw[dis_x : dis_x + top_platform_len, stair_y_lo:stair_y_hi] = current_height

        # Waypoint ON TOP of stairs
        waypoint_positions.append([dis_x + top_platform_len // 2, mid_y])
        waypoint_heights.append(current_height)
        dis_x += top_platform_len

        # Create stairs going down
        for i in range(cfg.num_steps):
            step_start = dis_x
            mark_edge_band(step_start - edge_band_pixels, step_start)
            current_height -= step_height_pixels
            height_field_raw[step_start : step_start + step_depth_pixels, stair_y_lo:stair_y_hi] = current_height
            dis_x += step_depth_pixels

        # Waypoint after the descending stairs.
        waypoint_positions.append([min(width_pixels - 1, dis_x + flat_spacing_pixels // 2), mid_y])
        waypoint_heights.append(0)

        # Flat spacing is geometry only between stair sets.
        if stair_set < cfg.num_stair_sets - 1:
            dis_x += flat_spacing_pixels

    # Final position - a bit forward from the last waypoint
    last_waypoint_x = waypoint_positions[-1][0]
    final_dis_x = last_waypoint_x + round(1.0 / cfg.horizontal_scale)
    final_dis_x = min(final_dis_x, width_pixels - round(0.5 / cfg.horizontal_scale))

    # Fill goals array:
    # goals[0] = start position
    # goals[1] to goals[num_waypoints] = collected waypoints
    # goals[num_waypoints+1] = final useful waypoint
    # remaining goals = exact copies of the final useful waypoint
    num_waypoints = len(waypoint_positions)
    first_final_goal = min(num_waypoints + 1, num_goals)

    goals = np.zeros((num_goals, 2))
    goal_heights = np.zeros(num_goals)

    # Start position
    goals[0] = [max(round(0.5 / cfg.horizontal_scale), first_stair_x - round(1.0 / cfg.horizontal_scale)), mid_y]
    goal_heights[0] = platform_height

    # Fill waypoints, then fill rest with final position
    for i in range(1, num_goals):
        if i - 1 < num_waypoints:
            goals[i] = waypoint_positions[i - 1]
            goal_heights[i] = waypoint_heights[i - 1]
        else:
            goals[i] = [final_dis_x, mid_y]
            goal_heights[i] = 0

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    # Compute per-waypoint max Y-shift range (constrained to stair width minus margin)
    stair_y_margin = 0.3  # meters margin to keep robot within stairs
    max_y_shift = max(0.0, stair_width / 2.0 - stair_y_margin)
    goal_y_shift_ranges = np.full(num_goals, max_y_shift)
    # The final useful goal and its unused copies must remain coincident after
    # per-environment goal randomization.
    goal_y_shift_ranges[first_final_goal:] = 0.0
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights, goal_y_shift_ranges, x_edge_mask


@parkour_field_to_mesh
def parkour_box_terrain(
    difficulty: float,
    cfg: parkour_terrains_cfg.ExtremeParkourBoxTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate narrow, step-like box platforms with randomized lateral centers."""
    box_step_height = eval(cfg.box_height, {"difficulty": difficulty})
    box_step_height_pixels = round(box_step_height / cfg.vertical_scale)

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels))

    mid_y = length_pixels // 2

    platform_len = round(cfg.platform_len / cfg.horizontal_scale)
    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    height_field_raw[0:platform_len, :] = platform_height

    goals = np.zeros((num_goals, 2))
    goal_heights = np.ones((num_goals)) * platform_height
    goals[0] = [platform_len - round(1 / cfg.horizontal_scale), mid_y]

    platform_width_range = (1.5, 3.0)
    platform_y_offset_range = (-0.2, 0.2)
    platform_depth_min = max(1, int(np.ceil(cfg.box_length_range[0] / cfg.horizontal_scale)))
    platform_depth_max = max(platform_depth_min, int(np.floor(cfg.box_length_range[1] / cfg.horizontal_scale)))
    platform_half_width_min = max(1, int(np.ceil(0.5 * platform_width_range[0] / cfg.horizontal_scale)))
    platform_half_width_max = max(
        platform_half_width_min,
        int(np.floor(0.5 * platform_width_range[1] / cfg.horizontal_scale)),
    )
    platform_y_offset_min = int(np.ceil(platform_y_offset_range[0] / cfg.horizontal_scale))
    platform_y_offset_max = int(np.floor(platform_y_offset_range[1] / cfg.horizontal_scale))
    center_y_min = platform_half_width_max
    center_y_max = length_pixels - platform_half_width_max
    if center_y_max < center_y_min:
        center_y_min = center_y_max = length_pixels // 2

    waypoint_y_shift_ranges = np.zeros(num_goals)
    x_edge_mask = np.zeros((width_pixels, length_pixels), dtype=bool)
    edge_band_pixels = max(1, round(0.15 / cfg.horizontal_scale))

    def mark_edge_band(x_lo: int, x_hi: int, y_span: tuple[int, int]):
        x_lo = max(0, x_lo)
        x_hi = min(width_pixels, x_hi)
        y_lo, y_hi = y_span
        y_lo = max(0, y_lo)
        y_hi = min(length_pixels, y_hi)
        if x_hi > x_lo and y_hi > y_lo:
            x_edge_mask[x_lo:x_hi, y_lo:y_hi] = True

    levels = [1, 2, 3, 2, 1]
    platform_depths = [np.random.randint(platform_depth_min, platform_depth_max + 1) for _ in levels]
    obstacle_length = sum(platform_depths)
    dis_x = max(0, (width_pixels - obstacle_length) // 2)

    center_y = mid_y
    prev_level = 0
    prev_y_span = None
    for i, (level, platform_depth_pixels) in enumerate(zip(levels, platform_depths)):
        half_width_pixels = np.random.randint(platform_half_width_min, platform_half_width_max + 1)
        platform_width = 2.0 * half_width_pixels * cfg.horizontal_scale

        center_y += np.random.randint(platform_y_offset_min, platform_y_offset_max + 1)
        center_y = int(np.clip(center_y, center_y_min, center_y_max))
        y_lo = max(0, center_y - half_width_pixels)
        y_hi = min(length_pixels, center_y + half_width_pixels)

        platform_start = dis_x
        platform_end = min(width_pixels, platform_start + platform_depth_pixels)
        if platform_end <= platform_start:
            break

        platform_height = level * box_step_height_pixels
        height_field_raw[platform_start:platform_end, y_lo:y_hi] = platform_height

        if level > prev_level:
            mark_edge_band(platform_start, platform_start + edge_band_pixels, (y_lo, y_hi))
        elif level < prev_level and prev_y_span is not None:
            mark_edge_band(platform_start - edge_band_pixels, platform_start, prev_y_span)

        goal_idx = i + 1
        goals[goal_idx] = [(platform_start + platform_end - 1) // 2, center_y]
        goal_heights[goal_idx] = platform_height
        waypoint_y_shift_ranges[goal_idx] = max(0.0, platform_width / 2.0 - 0.3)

        dis_x = platform_end
        prev_level = level
        prev_y_span = (y_lo, y_hi)

    if prev_level > 0 and prev_y_span is not None:
        mark_edge_band(dis_x - edge_band_pixels, dis_x, prev_y_span)

    border_pixels = int(cfg.border_width / cfg.horizontal_scale) + 1
    goals[0] = [-border_pixels, mid_y]
    final_dis_x = min(dis_x + 1.5 / cfg.horizontal_scale, width_pixels - round(0.5 / cfg.horizontal_scale))
    first_final_goal = len(levels) + 1
    goals[first_final_goal:] = [final_dis_x, center_y]
    goal_heights[first_final_goal:] = 0
    waypoint_y_shift_ranges[first_final_goal:] = 0.0

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    goal_y_shift_ranges = waypoint_y_shift_ranges
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights, goal_y_shift_ranges, x_edge_mask


@parkour_field_to_mesh
def parkour_pyramid_slope_terrain(
    difficulty: float,
    cfg: parkour_terrains_cfg.ExtremeParkourPyramidSlopeTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate IsaacLab-style pyramid slope terrain with parkour centerline waypoints."""
    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    if cfg.inverted:
        slope = -slope

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_max = int(slope * cfg.size[0] / 2.0 / cfg.vertical_scale)
    center_x = max(1, width_pixels // 2)
    center_y = max(1, length_pixels // 2)

    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)

    height_field_raw = height_max * xx * yy

    platform_half_width = int(cfg.platform_width / cfg.horizontal_scale / 2.0)
    x_pf = int(np.clip(width_pixels // 2 - platform_half_width, 0, width_pixels - 1))
    y_pf = int(np.clip(length_pixels // 2 - platform_half_width, 0, length_pixels - 1))
    platform_height = height_field_raw[x_pf, y_pf]
    height_field_raw = np.clip(height_field_raw, min(0, platform_height), max(0, platform_height))
    height_field_raw = np.rint(height_field_raw).astype(np.int16)

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    goals, goal_heights, goal_y_shift_ranges = compute_centerline_goals(height_field_raw, cfg, num_goals)
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights, goal_y_shift_ranges, None


@parkour_field_to_mesh
def parkour_wave_terrain(
    difficulty: float,
    cfg: parkour_terrains_cfg.ExtremeParkourWaveTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate IsaacLab-style wave terrain with parkour centerline waypoints."""
    num_waves = cfg.num_waves
    if cfg.num_waves_range is not None:
        num_waves = cfg.num_waves_range[0] + difficulty * (cfg.num_waves_range[1] - cfg.num_waves_range[0])
    if num_waves <= 0:
        raise ValueError(f"Number of waves must be positive. Got: {num_waves}.")

    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    amplitude_pixels = int(0.5 * amplitude / cfg.vertical_scale)

    wave_length = length_pixels / num_waves
    wave_number = 2.0 * np.pi / wave_length
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)

    height_field_raw = amplitude_pixels * (np.cos(yy * wave_number) + np.sin(xx * wave_number))
    height_field_raw = np.rint(height_field_raw).astype(np.int16)

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    goals, goal_heights, goal_y_shift_ranges = compute_centerline_goals(height_field_raw, cfg, num_goals)
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights, goal_y_shift_ranges, None


@parkour_field_to_mesh
def parkour_pyramid_stairs_terrain(
    difficulty: float,
    cfg: parkour_terrains_cfg.ExtremeParkourPyramidStairsTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one centered elongated pyramid staircase with zigzag waypoints."""
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    if cfg.inverted:
        step_height *= -1.0

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    step_width = cfg.step_width
    if cfg.step_width_range is not None:
        step_width = np.random.uniform(cfg.step_width_range[0], cfg.step_width_range[1])
    step_width_pixels = max(1, int(step_width / cfg.horizontal_scale))
    step_height_pixels = int(step_height / cfg.vertical_scale)
    platform_width_pixels = max(1, int(cfg.platform_width / cfg.horizontal_scale))

    height_field_raw = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    # x_edge_mask = np.zeros((width_pixels, length_pixels), dtype=bool) if difficulty > 0.4 else None
    x_edge_mask = None
    edge_band_pixels = 1
    pyramid_length_pixels = min(width_pixels, round(cfg.pyramid_length / cfg.horizontal_scale))
    pyramid_start_x = (width_pixels - pyramid_length_pixels) // 2
    pyramid_stop_x = pyramid_start_x + pyramid_length_pixels
    current_step_height = 0
    start_x = pyramid_start_x
    stop_x = pyramid_stop_x
    start_y = 0
    stop_y = length_pixels
    short_edge_pixels = min(stop_x - start_x, stop_y - start_y)
    num_steps = max(0, (short_edge_pixels - platform_width_pixels) // (2 * step_width_pixels))

    for _ in range(num_steps):
        start_x += step_width_pixels
        stop_x -= step_width_pixels
        start_y += step_width_pixels
        stop_y -= step_width_pixels
        if start_x >= stop_x or start_y >= stop_y:
            break

        if x_edge_mask is not None:
            if cfg.inverted:
                x_edge_mask[
                    max(pyramid_start_x, start_x - 1 - edge_band_pixels) : max(pyramid_start_x, start_x - 1),
                    start_y:stop_y,
                ] = True
                x_edge_mask[
                    min(pyramid_stop_x, stop_x + 1) : min(pyramid_stop_x, stop_x + 1 + edge_band_pixels),
                    start_y:stop_y,
                ] = True
                x_edge_mask[
                    start_x:stop_x,
                    max(0, start_y - 1 - edge_band_pixels) : max(0, start_y - 1),
                ] = True
                x_edge_mask[
                    start_x:stop_x,
                    min(length_pixels, stop_y + 1) : min(length_pixels, stop_y + 1 + edge_band_pixels),
                ] = True
            else:
                x_edge_mask[start_x + 1 : min(stop_x, start_x + 1 + edge_band_pixels), start_y:stop_y] = True
                x_edge_mask[max(start_x, stop_x - 1 - edge_band_pixels) : stop_x - 1, start_y:stop_y] = True
                x_edge_mask[start_x:stop_x, start_y + 1 : min(stop_y, start_y + 1 + edge_band_pixels)] = True
                x_edge_mask[start_x:stop_x, max(start_y, stop_y - 1 - edge_band_pixels) : stop_y - 1] = True

        current_step_height += step_height_pixels
        height_field_raw[start_x:stop_x, start_y:stop_y] = current_step_height

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    goals, goal_heights, goal_y_shift_ranges = compute_centerline_goals(height_field_raw, cfg, num_goals)
    mid_y = length_pixels // 2
    first_step_x = pyramid_start_x + step_width_pixels
    last_step_x = pyramid_stop_x - step_width_pixels
    goals[0] = [first_step_x // 2, mid_y]
    goals[-1] = [(last_step_x + width_pixels - 1) // 2, mid_y]
    goal_heights[0] = height_field_raw[int(goals[0, 0]), int(goals[0, 1])]
    goal_heights[-1] = height_field_raw[int(goals[-1, 0]), int(goals[-1, 1])]
    goal_y_shift_ranges[1:-1] = cfg.goal_y_shift_range
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights, goal_y_shift_ranges, x_edge_mask


@parkour_field_to_mesh
def parkour_discrete_obstacles_terrain(
    difficulty: float,
    cfg: parkour_terrains_cfg.ExtremeParkourDiscreteObstaclesTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate rotated boxes at waypoint candidates."""
    height_difficulty = difficulty**cfg.height_difficulty_power
    box_height = cfg.box_height_range[0] + height_difficulty * (
        cfg.box_height_range[1] - cfg.box_height_range[0]
    )
    max_cell_height = cfg.grid_height_range[0] + height_difficulty * (
        cfg.grid_height_range[1] - cfg.grid_height_range[0]
    )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_field_raw = np.zeros((width_pixels, length_pixels), dtype=np.int16)

    platform_height = round(cfg.platform_height / cfg.vertical_scale)
    mid_y = length_pixels // 2
    cell_size_x = max(1, round(cfg.grid_cell_size[0] / cfg.horizontal_scale))
    cell_size_y = max(1, round(cfg.grid_cell_size[1] / cfg.horizontal_scale))
    max_cell_height_pixels = max(0, round(max_cell_height / cfg.vertical_scale))

    for x_start in range(0, width_pixels, cell_size_x):
        x_end = min(x_start + cell_size_x, width_pixels)
        for y_start in range(0, length_pixels, cell_size_y):
            y_end = min(y_start + cell_size_y, length_pixels)
            cell_height = np.random.randint(0, max_cell_height_pixels + 1)
            height_field_raw[x_start:x_end, y_start:y_end] = platform_height + cell_height

    box_height_pixels = max(0, round(box_height / cfg.vertical_scale))

    goals = np.zeros((num_goals, 2))
    goal_heights = np.zeros(num_goals)
    goal_y_shift_ranges = np.zeros(num_goals)

    num_obstacle_rows = min(cfg.num_obstacle_rows, num_goals - 2)
    row_spacing = max(1, round(cfg.obstacle_row_spacing / cfg.horizontal_scale))
    obstacle_span = (num_obstacle_rows - 1) * row_spacing
    first_row_x = (width_pixels - 1 - obstacle_span) // 2
    obstacle_row_x_positions = first_row_x + np.arange(num_obstacle_rows) * row_spacing
    goal_before_x = max(0, first_row_x - row_spacing)
    goal_after_x = min(width_pixels - 1, int(obstacle_row_x_positions[-1]) + row_spacing)
    goal_clear_x = max(1, round(cfg.goal_clearance[0] / cfg.horizontal_scale))
    goal_clear_y = max(1, round(cfg.goal_clearance[1] / cfg.horizontal_scale))
    goal_shift_margin = max(0, round(0.2 * cfg.goal_y_shift_range / cfg.horizontal_scale))
    y_min = goal_clear_y + goal_shift_margin + 1
    y_max = length_pixels - goal_clear_y - goal_shift_margin - 1
    zigzag_offset = round(np.random.uniform(cfg.zigzag_y_range[0], cfg.zigzag_y_range[1]) / cfg.horizontal_scale)
    goal_y_shift_ranges[1 : num_obstacle_rows + 1] = 2.0 * zigzag_offset * cfg.horizontal_scale

    goals[0] = [goal_before_x, mid_y]
    for goal_idx, goal_px in enumerate(obstacle_row_x_positions, start=1):
        goal_px = int(goal_px)
        for direction in (-1, 1):
            goal_py = int(np.clip(mid_y + direction * zigzag_offset, y_min, y_max))
            if direction < 0:
                goals[goal_idx] = [goal_px, goal_py]
            if cfg.add_boxes:
                box_size = np.random.uniform(cfg.box_size_range[0], cfg.box_size_range[1]) / cfg.horizontal_scale
                box_angle = np.deg2rad(np.random.uniform(cfg.box_rotation_range[0], cfg.box_rotation_range[1]))
                fill_rotated_height_field_square(
                    height_field_raw,
                    goal_px,
                    goal_py,
                    box_size,
                    box_angle,
                    platform_height + box_height_pixels,
                )

    first_final_goal = num_obstacle_rows + 1
    goals[first_final_goal:] = [goal_after_x, mid_y]

    if cfg.add_boxes:
        clear_height_field_rect(
            height_field_raw,
            goal_after_x,
            mid_y,
            goal_clear_x,
            goal_clear_y,
            platform_height,
        )

    for goal_idx, goal in enumerate(goals):
        goal_px = int(np.clip(round(goal[0]), 0, width_pixels - 1))
        goal_py = int(np.clip(round(goal[1]), 0, length_pixels - 1))
        goal_heights[goal_idx] = height_field_raw[goal_px, goal_py]

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    return height_field_raw, goals * cfg.horizontal_scale, goal_heights, goal_y_shift_ranges, None


@parkour_field_to_mesh
def parkour_demo_terrain(
    difficulty: float,
    cfg: parkour_terrains_cfg.ExtremeParkourDemoTerrainCfg,
    num_goals: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    goals = np.zeros((num_goals, 2))
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    mid_y = length_pixels // 2  # length is actually y width

    height_field_raw = np.zeros((width_pixels, length_pixels))
    goal_heights = np.ones((num_goals)) * round(cfg.platform_height / cfg.vertical_scale)
    platform_length = round(2 / cfg.horizontal_scale)
    hurdle_depth = round(np.random.uniform(0.35, 0.4) / cfg.horizontal_scale)
    hurdle_height = round(np.random.uniform(0.3, 0.36) / cfg.vertical_scale)
    hurdle_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    goals[0] = [platform_length + hurdle_depth / 2, mid_y]
    height_field_raw[
        platform_length : platform_length + hurdle_depth,
        round(mid_y - hurdle_width / 2) : round(mid_y + hurdle_width / 2),
    ] = hurdle_height

    platform_length += round(np.random.uniform(1.5, 2.5) / cfg.horizontal_scale)
    first_step_depth = round(np.random.uniform(0.45, 0.8) / cfg.horizontal_scale)
    first_step_height = round(np.random.uniform(0.35, 0.45) / cfg.vertical_scale)
    first_step_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    goals[1] = [platform_length + first_step_depth / 2, mid_y]
    height_field_raw[
        platform_length : platform_length + first_step_depth,
        round(mid_y - first_step_width / 2) : round(mid_y + first_step_width / 2),
    ] = first_step_height
    goal_heights[1] = first_step_height

    platform_length += first_step_depth
    second_step_depth = round(np.random.uniform(0.45, 0.8) / cfg.horizontal_scale)
    second_step_height = first_step_height
    second_step_width = first_step_width
    goals[2] = [platform_length + second_step_depth / 2, mid_y]
    height_field_raw[
        platform_length : platform_length + second_step_depth,
        round(mid_y - second_step_width / 2) : round(mid_y + second_step_width / 2),
    ] = second_step_height
    goal_heights[2] = second_step_height

    # gap
    platform_length += second_step_depth
    gap_size = round(np.random.uniform(0.5, 0.8) / cfg.horizontal_scale)

    # step down
    platform_length += gap_size
    third_step_depth = round(np.random.uniform(0.25, 0.6) / cfg.horizontal_scale)
    third_step_height = first_step_height
    third_step_width = round(np.random.uniform(1, 1.2) / cfg.horizontal_scale)
    goals[3] = [platform_length + third_step_depth / 2, mid_y]
    height_field_raw[
        platform_length : platform_length + third_step_depth,
        round(mid_y - third_step_width / 2) : round(mid_y + third_step_width / 2),
    ] = third_step_height
    goal_heights[3] = third_step_height

    platform_length += third_step_depth
    forth_step_depth = round(np.random.uniform(0.25, 0.6) / cfg.horizontal_scale)
    forth_step_height = first_step_height
    forth_step_width = third_step_width
    goals[4] = [platform_length + forth_step_depth / 2, mid_y]
    height_field_raw[
        platform_length : platform_length + forth_step_depth,
        round(mid_y - forth_step_width / 2) : round(mid_y + forth_step_width / 2),
    ] = forth_step_height
    goal_heights[4] = forth_step_height

    # parkour
    platform_length += forth_step_depth
    gap_size = round(np.random.uniform(0.1, 0.4) / cfg.horizontal_scale)
    platform_length += gap_size

    left_y = mid_y + round(np.random.uniform(0.15, 0.3) / cfg.horizontal_scale)
    right_y = mid_y - round(np.random.uniform(0.15, 0.3) / cfg.horizontal_scale)

    slope_height = round(np.random.uniform(0.15, 0.22) / cfg.vertical_scale)
    slope_depth = round(np.random.uniform(0.75, 0.85) / cfg.horizontal_scale)
    slope_width = round(1.0 / cfg.horizontal_scale)

    platform_height = slope_height + np.random.randint(0, 0.2 / cfg.vertical_scale)

    goals[5] = [platform_length + slope_depth / 2, left_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * 1
    height_field_raw[
        platform_length : platform_length + slope_depth, left_y - slope_width // 2 : left_y + slope_width // 2
    ] = heights.astype(int) + platform_height
    goal_heights[5] = np.mean(heights.astype(int) + platform_height)

    platform_length += slope_depth + gap_size
    goals[6] = [platform_length + slope_depth / 2, right_y]
    heights = np.tile(np.linspace(-slope_height, slope_height, slope_width), (slope_depth, 1)) * -1
    height_field_raw[
        platform_length : platform_length + slope_depth, right_y - slope_width // 2 : right_y + slope_width // 2
    ] = heights.astype(int) + platform_height
    goal_heights[6] = np.mean(heights.astype(int) + platform_height)

    platform_length += slope_depth + gap_size + round(0.4 / cfg.horizontal_scale)
    goals[-1] = [platform_length, left_y]

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    goal_y_shift_ranges = np.zeros(num_goals)
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights * cfg.vertical_scale, goal_y_shift_ranges, None
