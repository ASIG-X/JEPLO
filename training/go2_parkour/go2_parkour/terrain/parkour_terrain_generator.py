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

import numpy as np
import omni.log
import trimesh
from go2_parkour.terrain.parkour_terrain_generator_cfg import ParkourSubTerrainBaseCfg, ParkourTerrainGeneratorCfg
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.trimesh.utils import make_border


class ParkourTerrainGenerator(TerrainGenerator):
    def __init__(self, cfg: ParkourTerrainGeneratorCfg, device: str = "cpu"):
        if cfg.sub_terrain_border_width is not None:
            for sub_cfg in cfg.sub_terrains.values():
                sub_cfg.border_width = cfg.sub_terrain_border_width

        self.num_goals = cfg.num_goals
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols))
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, self.num_goals, 3))
        self.terrain_names = np.zeros((cfg.num_rows, cfg.num_cols, 1)).astype(str)
        width_pixels = int(cfg.size[0] / cfg.horizontal_scale) + 1
        length_pixels = int(cfg.size[1] / cfg.horizontal_scale) + 1
        self.total_width_pixels = width_pixels * cfg.num_rows
        self.total_length_pixels = length_pixels * cfg.num_cols
        self.goal_heights = np.zeros((cfg.num_rows, cfg.num_cols, self.num_goals), dtype=np.int16)
        self.goal_y_shift_ranges = np.zeros((cfg.num_rows, cfg.num_cols, self.num_goals))
        self.x_edge_maskes = np.zeros((cfg.num_rows, cfg.num_cols, width_pixels, length_pixels), dtype=np.int16)
        self.pre_climb_maskes = np.zeros((cfg.num_rows, cfg.num_cols, width_pixels, length_pixels), dtype=np.int16)

        super().__init__(cfg=cfg, device=device)
        self.cfg: ParkourTerrainGeneratorCfg

        # Keep the terrain uniformly gray without highlighting edge-mask regions.
        self.terrain_mesh.visual.vertex_colors = np.full(
            (len(self.terrain_mesh.vertices), 4), [60, 60, 60, 255], dtype=np.uint8
        )

    def _generate_random_terrains(self):
        """Add terrains based on randomly sampled difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        sub_terrains_names = list(self.cfg.sub_terrains.keys())
        # randomly sample sub-terrains
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            # coordinate index of the sub-terrain
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            # randomly sample terrain index
            sub_index = self.np_rng.choice(len(proportions), p=proportions)
            # randomly sample difficulty parameter
            difficulty = self.np_rng.uniform(*self.cfg.difficulty_range)
            # generate terrain
            sub_terrains_name = sub_terrains_names[sub_index]
            self.terrain_type[sub_row, sub_col] = sub_col
            sub_terrains_cfg = sub_terrains_cfgs[sub_index]
            mesh, origin, sub_terrain_goal, goal_heights, x_edge_mask, pre_climb_mask, goal_y_shift_ranges = (
                self._get_terrain_mesh(difficulty, sub_terrains_cfg)
            )
            # add to sub-terrains
            self.terrain_names[sub_row, sub_col] = sub_terrains_name
            self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrain_goal)
            self.goal_heights[sub_row, sub_col, :] = goal_heights
            self.goal_y_shift_ranges[sub_row, sub_col, :] = goal_y_shift_ranges
            self.x_edge_maskes[sub_row, sub_col, :, :] = x_edge_mask
            self.pre_climb_maskes[sub_row, sub_col, :, :] = pre_climb_mask

    def _generate_curriculum_terrains(self):
        """Add terrains based on the difficulty parameter."""
        # normalize the proportions of the sub-terrains
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        # create a list of all terrain configs
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        sub_terrains_names = list(self.cfg.sub_terrains.keys())
        # curriculum-based sub-terrains
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                lower, upper = self.cfg.difficulty_range
                if self.cfg.random_difficulty:
                    difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                else:
                    difficulty = sub_row / (self.cfg.num_rows - 1)

                difficulty = lower + (upper - lower) * difficulty
                # generate terrain
                sub_terrains_cfg = sub_terrains_cfgs[sub_indices[sub_col]]
                sub_terrains_name = sub_terrains_names[sub_indices[sub_col]]
                mesh, origin, sub_terrain_goal, goal_heights, x_edge_mask, pre_climb_mask, goal_y_shift_ranges = (
                    self._get_terrain_mesh(difficulty, sub_terrains_cfg)
                )
                # add to sub-terrains
                self.terrain_type[sub_row, sub_col] = sub_indices[sub_col]
                self.terrain_names[sub_row, sub_col] = sub_terrains_name
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrain_goal)
                self.goal_heights[sub_row, sub_col, :] = goal_heights
                self.goal_y_shift_ranges[sub_row, sub_col, :] = goal_y_shift_ranges
                self.x_edge_maskes[sub_row, sub_col, :, :] = x_edge_mask
                self.pre_climb_maskes[sub_row, sub_col, :, :] = pre_climb_mask

    def _get_terrain_mesh(
        self,
        difficulty: float,
        cfg: ParkourSubTerrainBaseCfg,
    ) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
        # copy the configuration
        cfg: ParkourSubTerrainBaseCfg = cfg.copy()
        # add other parameters to the sub-terrain configuration
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        # generate hash for the sub-terrain
        # generate the terrain
        meshes, origin, goals, goal_heights, x_edge_mask, pre_climb_mask, goal_y_shift_ranges = cfg.function(
            difficulty, cfg, self.num_goals
        )
        mesh = trimesh.util.concatenate(meshes)
        # offset mesh such that they are in their center
        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        # change origin to be in the center of the sub-terrain
        origin += transform[0:3, -1]

        # if caching is enabled, save the mesh and origin

        return mesh, origin, goals, goal_heights, x_edge_mask, pre_climb_mask, goal_y_shift_ranges

    def _add_terrain_border(self):
        """Add a surrounding border over all the sub-terrains into the terrain meshes."""
        # border parameters
        border_size = (
            self.cfg.num_rows * self.cfg.size[0] + 2 * self.cfg.border_width,
            self.cfg.num_cols * self.cfg.size[1] + 2 * self.cfg.border_width,
        )
        inner_size = (self.cfg.num_rows * self.cfg.size[0], self.cfg.num_cols * self.cfg.size[1])
        border_center = (
            self.cfg.num_rows * self.cfg.size[0] / 2,
            self.cfg.num_cols * self.cfg.size[1] / 2,
            -self.cfg.border_height / 2,
        )
        # border mesh
        border_meshes = make_border(border_size, inner_size, height=self.cfg.border_height, position=border_center)
        border = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border.triangles)[:, :, 2] < -0.1).any(1)
        border.update_faces(selector)
        # add the border to the list of meshes
        self.terrain_meshes.append(border)

    def _add_sub_terrain(
        self,
        mesh: trimesh.Trimesh,
        origin: np.ndarray,
        row: int,
        col: int,
        sub_terrain_goal: np.ndarray,
    ):
        # transform the mesh to the correct position
        transform = np.eye(4)
        transform[0:2, -1] = (row + 0.5) * self.cfg.size[0], (col + 0.5) * self.cfg.size[1]
        mesh.apply_transform(transform)
        # add mesh to the list
        self.terrain_meshes.append(mesh)
        # add origin to the list
        self.terrain_origins[row, col] = origin + transform[:3, -1]
        self.goals[row, col, :, :2] = sub_terrain_goal

    def _color_edge_mask(self):
        """Paint x_edge_mask regions red on the combined terrain mesh.

        The terrain mesh has already been centered (offset by -total_size/2).
        We reverse that offset, determine which sub-terrain each vertex belongs
        to, compute local pixel indices, then map to the global mask.
        """
        # Build the full edge mask in pixel space: (total_width_pixels, total_length_pixels)
        # x_edge_maskes shape: (num_rows, num_cols, width_pixels, length_pixels)
        width_pixels = int(self.cfg.size[0] / self.cfg.horizontal_scale) + 1
        length_pixels = int(self.cfg.size[1] / self.cfg.horizontal_scale) + 1
        full_mask = self.x_edge_maskes.transpose(0, 2, 1, 3).reshape(self.total_width_pixels, self.total_length_pixels)

        # The mesh was centered: vertex_world = vertex_local - (total_size / 2)
        # So: vertex_local = vertex_world + (total_size / 2)
        offset_x = self.cfg.size[0] * self.cfg.num_rows * 0.5
        offset_y = self.cfg.size[1] * self.cfg.num_cols * 0.5
        h_scale = self.cfg.horizontal_scale
        sub_size_x = self.cfg.size[0]
        sub_size_y = self.cfg.size[1]

        verts = np.asarray(self.terrain_mesh.vertices)
        local_x = verts[:, 0] + offset_x
        local_y = verts[:, 1] + offset_y

        # Determine which sub-terrain row/col each vertex belongs to
        row = np.clip((local_x / sub_size_x).astype(int), 0, self.cfg.num_rows - 1)
        col = np.clip((local_y / sub_size_y).astype(int), 0, self.cfg.num_cols - 1)

        # Compute local pixel index within that sub-terrain
        local_px = np.clip(((local_x - row * sub_size_x) / h_scale).round().astype(int), 0, width_pixels - 1)
        local_py = np.clip(((local_y - col * sub_size_y) / h_scale).round().astype(int), 0, length_pixels - 1)

        # Map to global mask index
        px = row * width_pixels + local_px
        py = col * length_pixels + local_py

        # Look up edge mask
        is_edge = full_mask[px, py].astype(bool)

        # Assign vertex colors: gray default, red for edges
        colors = np.full((len(verts), 4), [60, 60, 60, 255], dtype=np.uint8)
        colors[is_edge] = [255, 0, 0, 255]
        self.terrain_mesh.visual.vertex_colors = colors
