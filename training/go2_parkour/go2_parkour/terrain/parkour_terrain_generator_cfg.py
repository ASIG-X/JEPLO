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

from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass


@configclass
class ParkourSubTerrainBaseCfg(HfTerrainBaseCfg):
    border_width: float = 0.0
    horizontal_scale: float = 0.08
    """The discretization of the terrain along the x and y axes (in m). Defaults to 0.1."""
    vertical_scale: float = 0.005
    """The discretization of the terrain along the z axis (in m). Defaults to 0.005."""
    platform_len: float = 2.5
    platform_height: float = 0.0
    slope_threshold: float | None = 1.5  # used for mesh vertex shifting only (visual quality)
    edge_width_thresh = 0.05
    use_simplified: bool = False


@configclass
class ParkourTerrainGeneratorCfg(TerrainGeneratorCfg):
    num_goals: int = 8
    sub_terrain_border_width: float | None = None
    terrain_names: list[str] = []
    random_difficulty: bool = False
