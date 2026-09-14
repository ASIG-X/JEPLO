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

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import math

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.assets import (
    Articulation,
)
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, RAY_CASTER_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.sensors import ContactSensor, RayCaster

from go2_parkour.sensor.mid360_raycaster import Mid360RayCaster
from go2_parkour.sensor.spherical_depth_generator import SphericalDepthGenerator
from go2_parkour.tasks.command_curriculum import (
    direct_command_curriculum_moves,
    update_direct_command_max_distance,
)
from go2_parkour.tasks.go2_loco_env_cfg import Go2LocoEnvCfg
from go2_parkour.tasks.go2_loco_rewards import (
    action_smoothness_l2,
    base_height,
    feet_regulation,
    hip_pos_penalty_l1,
    joint_pos_penalty_l1,
    linear_schedule,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
    undesired_contacts,
)
from go2_parkour.terrain.parkour_terrain_importer import ParkourTerrainImporter

DIRECT_COMMAND_TERRAIN_NAMES = (
    "parkour_flat",
    "parkour_pyramid_stairs",
    "parkour_inverted_pyramid_stairs",
    "parkour_discrete_grid",
)

POLICY_JOINT_NAMES = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]


class Go2LocoEnv(DirectRLEnv):
    cfg: Go2LocoEnvCfg

    def __init__(self, cfg: Go2LocoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        assert self._robot.joint_names == POLICY_JOINT_NAMES, (
            "Robot joint order does not match the policy joint order."
            f"\nExpected: {POLICY_JOINT_NAMES}"
            f"\nActual:   {self._robot.joint_names}"
        )

        self._reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Startup events have already randomized these quantities. Preserve the
        # existing privileged fields as truthful summaries of the applied values.
        base_body_ids, _ = self._robot.find_bodies("base")
        base_body_id = base_body_ids[0]
        masses = self._robot.root_physx_view.get_masses()
        default_base_mass = self._robot.data.default_mass[:, base_body_id : base_body_id + 1].to(masses.device)
        self._delta_masses = (
            masses[:, base_body_id : base_body_id + 1] - default_base_mass
        ).to(self.device)
        materials = self._robot.root_physx_view.get_material_properties()
        self._physics_parameters = materials.mean(dim=1).to(self.device)

        self._first_reset = True
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions = torch.zeros_like(self._actions)
        self._previous_previous_actions = torch.zeros_like(self._actions)
        self._num_actions = self._actions.shape[1]
        self._motor_zero_offset = torch.zeros_like(self._actions)

        self._obs_buffer = torch.zeros(
            self.num_envs, self.cfg.num_obs_hist, self.cfg.obs_proprio_dim, device=self.device
        )
        self._terminal_obs = torch.zeros(self.num_envs, self.cfg.observation_space, device=self.device)

        # body orders of self._contact_sensor and self._robot are different.
        self._base_id_cs, _ = self._contact_sensor.find_bodies(["base"])
        self._feet_ids_cs, _ = self._contact_sensor.find_bodies(".*foot")
        self._undesired_contact_body_ids_cs, _ = self._contact_sensor.find_bodies([".*thigh", ".*calf"])
        self._feet_ids_bd, _ = self._robot.find_bodies(".*foot")
        self._hip_ids_jt, _ = self._robot.find_joints(".*hip.*")
        self._thigh_calf_ids_jt, _ = self._robot.find_joints(".*(thigh|calf).*")

        self._reward_weights = {
            "track_lin_vel_xy_exp": self.cfg.rewards.track_lin_vel_xy_exp,
            "track_ang_vel_z_exp": self.cfg.rewards.track_ang_vel_z_exp,
            "lin_vel_z_l2": self.cfg.rewards.lin_vel_z_l2_initial,
            "ang_vel_xy_l2": self.cfg.rewards.ang_vel_xy_l2,
            "joint_acc_l2": self.cfg.rewards.joint_acc_l2,
            "joint_power": self.cfg.rewards.joint_power,
            "joint_torques_l2": self.cfg.rewards.joint_torques_l2,
            "base_height_l2": self.cfg.rewards.base_height_l2_initial,
            "action_rate_l2": self.cfg.rewards.action_rate_l2,
            "action_smoothness_l2": self.cfg.rewards.action_smoothness_l2,
            "undesired_contacts": self.cfg.rewards.undesired_contacts,
            "joint_pos_limits": self.cfg.rewards.joint_pos_limits,
            "feet_regulation": self.cfg.rewards.feet_regulation,
            "hip_pos_penalty_l1": self.cfg.rewards.hip_pos_penalty_l1,
            "joint_pos_penalty_l1": self.cfg.rewards.joint_pos_penalty_l1,
        }
        self._episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for name in self._reward_weights
        }
        # commands
        self._cmd_lin_vel = torch.zeros(self.num_envs, 2, device=self.device)
        self._cmd_ang_vel = torch.zeros(self.num_envs, 1, device=self.device)
        self._cmd_resample_intervals = torch.zeros(self.num_envs, 1, device=self.device)
        self._cmd_resample_accums = torch.zeros(self.num_envs, 1, device=self.device)
        self._cmd_commands_low = torch.tensor([0.5, 0.5, -2.0], device=self.device)
        self._cmd_commands_high = torch.tensor([1.0, 1.0, 2.0], device=self.device)
        self._cmd_not_zero_out_prob = torch.tensor([0.9, 0.75, 0.5], device=self.device)
        self._cmd_goal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_goal_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_speed = torch.zeros(self.num_envs, 1, device=self.device)
        self._cmd_in_place_rotation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._cmd_in_place_ang_vel = torch.zeros(self.num_envs, device=self.device)
        self._commands_xy_accumulation = torch.zeros(self.num_envs, 2, device=self.device)
        self._max_move_distance = torch.zeros(self.num_envs, device=self.device)
        self._cmd_heading_offset = torch.zeros(self.num_envs, device=self.device)  # omni-directional on flat terrain

        self._spherical_scan_clip_distance = 3.0
        self._invalidate_prob_scale = torch.ones(self.num_envs, device=self.device)

        # -- setup terrain goals --
        terrain_generator = self._terrain.terrain_generator_class
        self._terrain_names = terrain_generator.terrain_names
        direct_terrain_cells_np = np.isin(self._terrain_names[:, :, -1], DIRECT_COMMAND_TERRAIN_NAMES)
        terrain_generator.goals[direct_terrain_cells_np] = 0.0
        terrain_generator.goal_heights[direct_terrain_cells_np] = 0
        terrain_generator.goal_y_shift_ranges[direct_terrain_cells_np] = 0.0

        self._terrain_goals = torch.from_numpy(terrain_generator.goals).to(self.device).to(torch.float32)
        self._terrain_goals += self._terrain.terrain_origins[:, :, None, :]

        terrain_goal_heights = terrain_generator.goal_heights * self.cfg.terrain.terrain_generator.vertical_scale
        self._terrain_goals[:, :, :, 2] = torch.from_numpy(terrain_goal_heights).to(self.device).to(torch.float32)
        direct_terrain_cells = torch.from_numpy(direct_terrain_cells_np).to(self.device)
        self._terrain_goals[direct_terrain_cells] = 0.0

        # -- per-waypoint lateral shift ranges (Y-axis) for non-flat terrains --
        self._terrain_goal_y_shift_ranges = (
            torch.from_numpy(terrain_generator.goal_y_shift_ranges).to(self.device).to(torch.float32)
        )  # shape: (num_rows, num_cols, num_goals)

        self._env_next_goals_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._env_goals = self._terrain_goals[self._terrain.terrain_levels, self._env_terrain_cols, :, :]

        # Apply initial random Y-shifts
        self._apply_goal_y_shifts(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))

        print("terrain goals shape:", self._terrain_goals.shape)
        print("env goals shape:", self._env_goals.shape)

        self._cmd_goal_w = self._env_goals[torch.arange(self.num_envs), self._env_next_goals_idx, :]

        self._env_terrain_names = self._terrain_names[
            self._terrain.terrain_levels.cpu().numpy(), self._env_terrain_cols.cpu().numpy()
        ]
        self._non_flat_terrain_mask = torch.from_numpy(self._env_terrain_names != "parkour_flat")[:, -1]
        self._flat_terrain_mask = ~self._non_flat_terrain_mask

        self._setup_rewards()

        self._occlusion_level = math_utils.sample_uniform(0.0, 0.4, self.num_envs, device=self.device)
        self._spherical_depth_gen = SphericalDepthGenerator(
            scanner=self._mid360,
            robot=self._robot,
            num_envs=self.num_envs,
            device=self.device,
            num_stacked=self.cfg.lidar_num_stacked,
            update_freq=self.cfg.depth_update_freq,
            noise_std=self.cfg.lidar_noise_std,
            bottom_dropout_fraction=self.cfg.lidar_bottom_dropout_fraction,
            occlusion_level=self._occlusion_level,
        )

        self._depth_buffer = torch.zeros(
            self.num_envs,
            self.cfg.num_depth_stack + self.cfg.depth_delay,
            self.cfg.mid360.pattern_cfg.vertical_num_rays,
            self.cfg.mid360.pattern_cfg.horizontal_num_rays,
            device=self.device,
        )

        self._height_map_shape = (
            int(
                math.ceil(self.cfg.height_scanner.pattern_cfg.size[0] / self.cfg.height_scanner.pattern_cfg.resolution)
            ),
            int(
                math.ceil(self.cfg.height_scanner.pattern_cfg.size[1] / self.cfg.height_scanner.pattern_cfg.resolution)
            ),
        )
        print(f"Height map shape: {self._height_map_shape}")

        # -- setup debug visualization --
        self._show_debug_viz = False
        # self.cfg.is_play_env = True
        if self.cfg.is_play_env:
            print("\n********************************************")
            print("           **Enabling debug draw.**")
            print("********************************************\n")
            self._show_debug_viz = True
            # fmt: off
            import isaacsim
            from isaacsim.core.utils.extensions import enable_extension
            enable_extension("omni.isaac.debug_draw")
            from isaacsim.util.debug_draw import _debug_draw
            # fmt: on
            self._debug_draw = _debug_draw.acquire_debug_draw_interface()

            goal_vel_viz_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
            cur_vel_viz_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_current")
            goal_vel_viz_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
            cur_vel_viz_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

            self._goal_vel_viz = VisualizationMarkers(cfg=goal_vel_viz_cfg)
            self._cur_vel_viz = VisualizationMarkers(cfg=cur_vel_viz_cfg)
            self._goal_vel_viz.set_visibility(True)
            self._cur_vel_viz.set_visibility(True)

            # goal_viz_cfg = copy.deepcopy(RAY_CASTER_MARKER_CFG)
            # goal_viz_cfg.prim_path = "/Visuals/GoalPoints"
            # goal_viz_cfg.markers["hit"].radius = 0.2
            # goal_viz_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            # self._goal_viz = VisualizationMarkers(cfg=goal_viz_cfg)
            # self._goal_viz.set_visibility(True)

            # mid360_viz_cfg = copy.deepcopy(RAY_CASTER_MARKER_CFG)
            # mid360_viz_cfg.prim_path = "/Visuals/Mid360Pos"
            # mid360_viz_cfg.markers["hit"].radius = 0.01
            # mid360_viz_cfg.markers["hit"].visual_material.diffuse_color = (1.0, 0.5, 0.0)
            # self._mid360_viz = VisualizationMarkers(cfg=mid360_viz_cfg)
            # self._mid360_viz.set_visibility(True)

            # pointcloud_viz_cfg = copy.deepcopy(RAY_CASTER_MARKER_CFG)
            # pointcloud_viz_cfg.prim_path = "/Visuals/PointCloud"
            # pointcloud_viz_cfg.markers["hit"].radius = 0.005
            # pointcloud_viz_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 0.5, 1.0)
            # self._pointcloud_viz = VisualizationMarkers(cfg=pointcloud_viz_cfg)
            # self._pointcloud_viz.set_visibility(True)

            height_map_viz_cfg = copy.deepcopy(RAY_CASTER_MARKER_CFG)
            height_map_viz_cfg.prim_path = "/Visuals/HeightMap"
            height_map_viz_cfg.markers["hit"].radius = 0.01
            height_map_viz_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 1.0)
            self._height_map_viz = VisualizationMarkers(cfg=height_map_viz_cfg)
            self._height_map_viz.set_visibility(True)

            # spherical_scan_viz_cfg = copy.deepcopy(RAY_CASTER_MARKER_CFG)
            # spherical_scan_viz_cfg.prim_path = "/Visuals/SphericalScan"
            # spherical_scan_viz_cfg.markers["hit"].radius = 0.01
            # spherical_scan_viz_cfg.markers["hit"].visual_material.diffuse_color = (1.0, 0.7, 0.0)
            # self._spherical_scan_viz = VisualizationMarkers(cfg=spherical_scan_viz_cfg)
            # self._spherical_scan_viz.set_visibility(True)

            # projected_spherical_scan_viz_cfg = copy.deepcopy(RAY_CASTER_MARKER_CFG)
            # projected_spherical_scan_viz_cfg.prim_path = "/Visuals/ProjectedSphericalScan"
            # projected_spherical_scan_viz_cfg.markers["hit"].radius = 0.01
            # projected_spherical_scan_viz_cfg.markers["hit"].visual_material.diffuse_color = (1.0, 0.0, 1.0)
            # self._projected_spherical_scan_viz = VisualizationMarkers(cfg=projected_spherical_scan_viz_cfg)
            # self._projected_spherical_scan_viz.set_visibility(True)

            env_goals_viz_cfg = copy.deepcopy(RAY_CASTER_MARKER_CFG)
            env_goals_viz_cfg.prim_path = "/Visuals/EnvGoals"
            env_goals_viz_cfg.markers["hit"].radius = 0.2
            env_goals_viz_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            self._env_goals_viz = VisualizationMarkers(cfg=env_goals_viz_cfg)
            self._env_goals_viz.set_visibility(True)

            env_next_goals_viz_cfg = copy.deepcopy(RAY_CASTER_MARKER_CFG)
            env_next_goals_viz_cfg.prim_path = "/Visuals/EnvNextGoals"
            env_next_goals_viz_cfg.markers["hit"].radius = 0.2
            env_next_goals_viz_cfg.markers["hit"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
            self._env_next_goals_viz = VisualizationMarkers(cfg=env_next_goals_viz_cfg)
            self._env_next_goals_viz.set_visibility(True)

            # Data collection counter (collect every 3rd call for ~17Hz from 50Hz)
            self._data_collect_counter = 0

        # predicted height map buffer (set externally via set_predicted_height_map)
        self._pred_height_map = None

        if not hasattr(self, "_track_env_id"):
            self.initialize_playback_controls()

        self._update_debug_draw()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # # do we need this in simulation?
        # self._setup_robot_ray_caster_proxy()

        # compute terrain sizes
        self.num_terrain_rows = self.cfg.terrain.terrain_generator.num_rows
        self.num_terrain_cols = self.cfg.terrain.terrain_generator.num_cols
        print(f"Terrain rows: {self.num_terrain_rows}, cols: {self.num_terrain_cols}")

        # create terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.cfg.terrain.terrain_generator.num_rows = self.num_terrain_rows
        self.cfg.terrain.terrain_generator.num_cols = self.num_terrain_cols
        self._terrain: ParkourTerrainImporter = self.cfg.terrain.class_type(self.cfg.terrain)

        # Assign every prefix of environment IDs according to the configured terrain proportions.
        sub_terrains = self.cfg.terrain.terrain_generator.sub_terrains
        terrain_order = list(sub_terrains)
        proportions = np.asarray([sub_terrains[name].proportion for name in terrain_order], dtype=np.float64)
        proportions /= proportions.sum()

        column_terrain_names = np.asarray(self._terrain.terrain_generator_class.terrain_names)[0, :, -1]
        columns_by_terrain = [np.flatnonzero(column_terrain_names == name) for name in terrain_order]
        for terrain_name, proportion, columns in zip(terrain_order, proportions, columns_by_terrain):
            if proportion > 0.0 and len(columns) == 0:
                raise RuntimeError(f"No terrain column was generated for subterrain '{terrain_name}'.")

        assigned_counts = np.zeros(len(terrain_order), dtype=np.int64)
        env_terrain_cols = np.empty(self.num_envs, dtype=np.int64)
        for env_id in range(self.num_envs):
            deficits = (env_id + 1) * proportions - assigned_counts
            terrain_idx = int(np.argmax(deficits))
            terrain_columns = columns_by_terrain[terrain_idx]
            env_terrain_cols[env_id] = terrain_columns[assigned_counts[terrain_idx] % len(terrain_columns)]
            assigned_counts[terrain_idx] += 1

        self._env_terrain_cols = torch.as_tensor(env_terrain_cols, device=self.device)
        self._terrain.env_origins[:] = self._terrain.terrain_origins[0, self._env_terrain_cols]
        self._terrain.terrain_levels[:] = 0
        if self.cfg.is_play_env:
            self._ui_terrain_levels = self._terrain.terrain_levels.clone()

        # create sensors
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        self._height_scanner_small = RayCaster(self.cfg.height_scanner_small)
        self.scene.sensors["height_scanner_small"] = self._height_scanner_small

        self._mid360 = Mid360RayCaster(self.cfg.mid360)
        self.scene.sensors["mid360"] = self._mid360

        self._height_scan_shape = (
            math.ceil(self.cfg.height_scanner.pattern_cfg.size[0] / self.cfg.height_scanner.pattern_cfg.resolution),
            math.ceil(self.cfg.height_scanner.pattern_cfg.size[1] / self.cfg.height_scanner.pattern_cfg.resolution),
        )
        print("Height scan shape:", self._height_scan_shape)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # lights
        sky_light_cfg = sim_utils.DomeLightCfg(intensity=1000.0)
        sky_light_cfg.func("/World/skyLight", sky_light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_previous_actions.copy_(self._previous_actions)
        self._previous_actions.copy_(self._actions)
        self._actions = actions.clone()

        # reset commands
        self._cmd_resample_accums += self.step_dt
        self._reset_speed(self._cmd_resample_accums >= self._cmd_resample_intervals)
        if self.cfg.is_play_env and torch.any(self._ui_in_place_rotation):
            self._cmd_in_place_rotation[self._ui_in_place_rotation] = True
            self._cmd_in_place_ang_vel[self._ui_in_place_rotation] = self._ui_in_place_ang_vel[
                self._ui_in_place_rotation
            ]

        update_direct_command_max_distance(
            self._max_move_distance,
            self._robot.data.root_pos_w[:, :2],
            self._terrain.env_origins[:, :2],
            self._direct_command_terrain_mask,
        )

        # -- update goals if current goal reached --
        dist_to_goals = torch.norm(self._cmd_goal_w[:, :2] - self._robot.data.root_pos_w[:, :2], dim=1)
        goal_reached = (dist_to_goals < 0.5) & self._waypoint_command_terrain_mask
        self._env_next_goals_idx[goal_reached] += 1
        clipped_next_goals_idx = self._env_next_goals_idx.clip(min=0, max=self._terrain_goals.shape[2] - 1)
        self._cmd_goal_w[goal_reached] = self._env_goals[torch.arange(self.num_envs), clipped_next_goals_idx, :][
            goal_reached
        ]

        # compute goal in body frame
        self._cmd_goal_b = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self._robot.data.root_quat_w), self._cmd_goal_w - self._robot.data.root_pos_w
        )
        goal_dir = self._cmd_goal_b[:, :2] / (torch.norm(self._cmd_goal_b[:, :2], dim=1, keepdim=True) + 1e-6)

        # Omni-directional heading on flat terrain:
        # The desired heading in world frame = goal_direction_world + heading_offset,
        # so the robot faces a random direction while still navigating toward the goal.
        # In body frame this means the desired yaw error = atan2(goal_dir) + heading_offset.
        goal_angle_b = torch.atan2(goal_dir[:, 1], goal_dir[:, 0])  # angle to goal in body frame
        desired_yaw_error = math_utils.wrap_to_pi(goal_angle_b + self._cmd_heading_offset)
        desired_yaw_rate = desired_yaw_error.clip(min=-1.0, max=1.0)

        self._cmd_lin_vel[self._waypoint_command_terrain_mask, :] = (
            goal_dir[self._waypoint_command_terrain_mask] * self._cmd_speed[self._waypoint_command_terrain_mask]
        )
        self._cmd_ang_vel[self._waypoint_command_terrain_mask, 0] = desired_yaw_rate[
            self._waypoint_command_terrain_mask
        ]

        dist_to_goal = torch.norm(self._cmd_goal_b[:, :2], dim=1)
        stopped_at_goal = (dist_to_goal < 0.5) & self._waypoint_command_terrain_mask
        self._cmd_lin_vel[stopped_at_goal, :] = 0.0
        self._cmd_ang_vel[stopped_at_goal, :] = 0.0

        self._cmd_lin_vel[self._cmd_in_place_rotation, :] = 0.0
        self._cmd_ang_vel[self._cmd_in_place_rotation, 0] = self._cmd_in_place_ang_vel[self._cmd_in_place_rotation]

        self._update_debug_draw()

        if self.cfg.is_play_env:
            self._update_tracking_camera()

    def _apply_action(self):
        joint_pos_target = (
            self._actions * self.cfg.action_scale + self._robot.data.default_joint_pos + self._motor_zero_offset
        )
        joint_pos_target = joint_pos_target.clamp(*self.cfg.action_target_clip)
        self._robot.set_joint_position_target(joint_pos_target)

    @property
    def tracked_env_id(self) -> int:
        return self._track_env_id

    def initialize_playback_controls(self) -> None:
        """Initialize camera and UI state used by the playback controls."""

        self._track_env_id = 0
        self._track_camera = True
        self._tracking_camera_offset = torch.tensor((-1.0, -2.0, 1.0), device=self.device)
        self._ui_in_place_rotation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ui_in_place_ang_vel = torch.zeros(self.num_envs, device=self.device)
        if not hasattr(self, "_ui_terrain_levels"):
            self._ui_terrain_levels = self._terrain.terrain_levels.clone()

    def cycle_tracked_env(self, offset: int) -> None:
        """Select another environment for camera tracking and UI controls."""

        self._track_env_id = (self._track_env_id + offset) % self.num_envs
        print(f"Now tracking env {self._track_env_id}")

    def reset_tracked_env(self) -> None:
        """Request a reset of the tracked environment on the next step."""

        self.episode_length_buf[self._track_env_id] = self.max_episode_length
        print(f"Reset env {self._track_env_id}")

    def change_tracked_terrain_level(self, delta: int) -> None:
        """Change the tracked environment terrain level and request a reset."""

        env_id = self._track_env_id
        level = int(self._ui_terrain_levels[env_id].item())
        level = max(0, min(self.num_terrain_rows - 1, level + delta))
        self._ui_terrain_levels[env_id] = level
        self.reset_tracked_env()
        print(f"Env {env_id} terrain level: {level}")

    def toggle_camera_tracking(self) -> None:
        self._track_camera = not self._track_camera
        print(f"Track camera: {self._track_camera}")

    def toggle_tracked_in_place_rotation(self) -> None:
        """Immediately toggle the tracked environment's in-place command override."""

        env_id = self._track_env_id
        enabled = not bool(self._ui_in_place_rotation[env_id].item())
        self._ui_in_place_rotation[env_id] = enabled
        if enabled:
            yaw_velocity = math_utils.sample_uniform(-2.0, 2.0, 1, self.device).squeeze()
            self._ui_in_place_ang_vel[env_id] = yaw_velocity
            self._cmd_in_place_rotation[env_id] = True
            self._cmd_in_place_ang_vel[env_id] = yaw_velocity
            self._cmd_lin_vel[env_id] = 0.0
            self._cmd_ang_vel[env_id, 0] = yaw_velocity
            print(f"Env {env_id} in-place rotation: True (wz={yaw_velocity.item():.3f})")
        else:
            self._ui_in_place_ang_vel[env_id] = 0.0
            self._cmd_in_place_rotation[env_id] = False
            self._cmd_in_place_ang_vel[env_id] = 0.0
            self._cmd_resample_accums[env_id] = self._cmd_resample_intervals[env_id]
            print(f"Env {env_id} in-place rotation: False")

    def _update_tracking_camera(self) -> None:
        if not self._track_camera or self.viewport_camera_controller is None:
            return

        lookat = self._robot.data.root_pos_w[self._track_env_id]
        eye = math_utils.quat_apply_yaw(
            self._robot.data.root_quat_w[self._track_env_id].unsqueeze(0),
            self._tracking_camera_offset.unsqueeze(0),
        )[0]
        eye += lookat
        self.viewport_camera_controller.update_view_location(
            eye=eye.detach().cpu().tolist(),
            lookat=lookat.detach().cpu().tolist(),
        )

    def set_predicted_height_map(self, pred_height_map: torch.Tensor):
        """Store predicted height map from world model for visualization.

        Args:
            pred_height_map: (num_envs, obs_height_scan_dim) predicted height map tensor.
        """
        self._pred_height_map = pred_height_map

    def get_sensor_size(self):
        pcfg = self.cfg.mid360.pattern_cfg
        return (pcfg.vertical_num_rays, pcfg.horizontal_num_rays, self.cfg.num_depth_stack)

    def get_height_map_size(self):
        return self._height_map_shape

    def get_terminal_observations(self):
        return self._terminal_obs

    def _get_observations(self, pure: bool = False) -> dict:
        # feet contact indicators
        feet_contact = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs, :], dim=-1) > 0.1
        feet_contact = feet_contact.float()

        # body contact forces
        contact_forces = torch.norm(self._contact_sensor.data.net_forces_w[:, :, :], dim=-1)  # (num_envs, 19)

        scan_hits_w = self._height_scanner.data.ray_hits_w
        height_data = scan_hits_w[:, :, 2] - self._robot.data.root_pos_w[:, 2:3]
        height_data = height_data.nan_to_num_(nan=2.0, posinf=2.0, neginf=-2.0)
        height_data = height_data.clip(min=-2.0, max=2.0) * 0.5

        # -- compute per-foot clearance above terrain --
        foot_pos_w = self._robot.data.body_pos_w[:, self._feet_ids_bd, :]  # (num_envs, 4, 3)
        xy_dist = torch.norm(
            foot_pos_w[:, :, None, :2] - scan_hits_w[:, None, :, :2], dim=-1
        )  # (num_envs, 4, num_rays)
        nearest_idx = xy_dist.argmin(dim=-1)  # (num_envs, 4)
        terrain_z = torch.gather(scan_hits_w[:, :, 2].nan_to_num(nan=-100.0), dim=1, index=nearest_idx)  # (num_envs, 4)
        if not pure:
            self._foot_clearance = (foot_pos_w[:, :, 2] - terrain_z).clamp(min=0.0)  # (num_envs, 4)

        # update lidar spherical depth buffer
        self._spherical_depth_gen.update(self.common_step_counter)
        if self.common_step_counter % self.cfg.depth_update_freq == 0:
            depth_image = self._spherical_depth_gen.get_depth_image()  # (N, V, H) in [0, 1]
            self._depth_buffer = torch.cat([self._depth_buffer[:, 1:, :, :], depth_image[:, None, :, :]], dim=1)

        if (
            not pure
            and self.cfg.is_play_env
            and self._window is not None
            and hasattr(self._window, "update_sensor_images")
        ):
            self._window.update_sensor_images(height_data)

        # build proprioception buffer
        flat_terrain = self._flat_terrain_mask.to(device=self.device)
        terrain_flat_one_hot = torch.stack([flat_terrain, ~flat_terrain], dim=-1).float()
        proprio_buf = torch.cat(
            [
                # proprioception
                self._robot.data.root_com_ang_vel_b * 0.25,  # 3
                self._robot.data.projected_gravity_b,  # 3
                self._cmd_lin_vel * 2.0,  # 2
                self._cmd_ang_vel * 0.25,  # 1
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # 12
                self._robot.data.joint_vel * 0.05,  # 12
                self._actions * 0.1,  # 12
            ],
            dim=-1,
        )
        noise_buf = torch.cat(
            [
                torch.ones(3) * 0.05,  # angular velocity
                torch.ones(3) * 0.05,  # projected gravity
                torch.zeros(3),  # commands
                torch.ones(12) * 0.01,  # joint positions
                torch.ones(12) * 0.01,  # joint velocities
                torch.zeros(self._num_actions),  # actions
            ],
            dim=0,
        )
        noisy_prorpio_buf = proprio_buf + (torch.rand_like(proprio_buf) * 2.0 - 1.0) * noise_buf.to(self.device)

        obs_buffer = torch.cat([noisy_prorpio_buf[:, None, :], self._obs_buffer[:, :-1, :]], dim=1)
        if not pure:
            # update observation history buffer before constructing final obs buffer
            self._obs_buffer = torch.cat([noisy_prorpio_buf[:, None, :], self._obs_buffer[:, :-1, :]], dim=1)

        obs_buf = torch.cat(
            [
                # proprioception history
                obs_buffer.reshape(self.num_envs, -1),
                # height scan
                height_data,
                # privileged information
                self._robot.data.root_com_lin_vel_b,  # 3
                self._foot_clearance,  # 4
                feet_contact,  # 4
                terrain_flat_one_hot,  # 2
                contact_forces * 0.01,  # 19
                self._robot.data.applied_torque * 0.01,  # 12
                self._robot.actuators["GO2HV"].stiffness[:, 0:1] * 0.01,  # 1
                self._robot.actuators["GO2HV"].damping[:, 0:1] * 0.25,  # 1
                self._physics_parameters,  # 3
                self._delta_masses * 0.2,  # 1
                # noise-free current proprioception for the critic only
                proprio_buf,  # 45
            ],
            dim=-1,
        )

        # Depth images returned separately to avoid bloating the PPO observation buffer
        depth_buf = self._depth_buffer[:, : self.cfg.num_depth_stack, :, :]

        if obs_buf.shape[1] != self.cfg.observation_space:
            raise RuntimeError(f"Expected obs: {self.cfg.observation_space}, got {obs_buf.shape[1]}")

        return {"policy": obs_buf, "depth": depth_buf}

    def _setup_rewards(self):
        """Initialize terrain masks shared by commands and curriculum."""
        self._terrain_names = self._terrain.terrain_generator_class.terrain_names
        self._env_terrain_names = self._terrain_names[
            self._terrain.terrain_levels.cpu().numpy(), self._env_terrain_cols.cpu().numpy()
        ]
        self._non_flat_terrain_mask = torch.from_numpy(self._env_terrain_names != "parkour_flat")[:, -1].to(self.device)
        self._gap_terrain_mask = torch.from_numpy(self._env_terrain_names[:, -1] == "parkour_gap").to(self.device)
        self._reduced_lin_vel_z_terrain_mask = torch.from_numpy(
            np.isin(self._env_terrain_names[:, -1], ("parkour_box", "parkour_discrete_obstacles", "parkour_gap"))
        ).to(self.device)
        # self._stairs_terrain_mask = torch.from_numpy(
        #     self._env_terrain_names[:, -1] == "parkour_stairs"
        # ).to(self.device)
        self._non_flat_direct_command_terrain_mask = torch.from_numpy(
            np.isin(self._env_terrain_names[:, -1], DIRECT_COMMAND_TERRAIN_NAMES[1:])
        ).to(self.device)
        self._flat_terrain_mask = ~self._non_flat_terrain_mask
        self._direct_command_terrain_mask = self._flat_terrain_mask | self._non_flat_direct_command_terrain_mask
        self._waypoint_command_terrain_mask = ~self._direct_command_terrain_mask

    def _get_rewards(self) -> torch.Tensor:
        command = torch.cat([self._cmd_lin_vel, self._cmd_ang_vel], dim=-1)
        measured_base_height = base_height(
            self._robot.data.root_pos_w,
            self._height_scanner_small.data.ray_hits_w,
            self.cfg.rewards.base_height_target,
        )

        out_of_limits = -(
            self._robot.data.joint_pos - self._robot.data.soft_joint_pos_limits[:, :, 0]
        ).clip(max=0.0)
        out_of_limits += (
            self._robot.data.joint_pos - self._robot.data.soft_joint_pos_limits[:, :, 1]
        ).clip(min=0.0)

        reward_rates = {
            "track_lin_vel_xy_exp": track_lin_vel_xy_exp(command, self._robot.data.root_com_lin_vel_b),
            "track_ang_vel_z_exp": track_ang_vel_z_exp(command, self._robot.data.root_ang_vel_b),
            "lin_vel_z_l2": torch.square(self._robot.data.root_lin_vel_b[:, 2]),
            "ang_vel_xy_l2": torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1),
            "joint_acc_l2": torch.sum(torch.square(self._robot.data.joint_acc), dim=1),
            "joint_power": torch.sum(
                torch.abs(self._robot.data.joint_vel * self._robot.data.applied_torque), dim=1
            ),
            "joint_torques_l2": torch.sum(torch.square(self._robot.data.applied_torque), dim=1),
            "base_height_l2": torch.square(measured_base_height - self.cfg.rewards.base_height_target)
            * (~self._gap_terrain_mask),
            "action_rate_l2": torch.sum(torch.square(self._actions - self._previous_actions), dim=1),
            "action_smoothness_l2": action_smoothness_l2(
                self._actions,
                self._previous_actions,
                self._previous_previous_actions,
            ),
            "undesired_contacts": undesired_contacts(
                self._contact_sensor.data.net_forces_w_history[:, :, self._undesired_contact_body_ids_cs],
                self.cfg.rewards.undesired_contact_threshold,
            ),
            "joint_pos_limits": torch.sum(out_of_limits, dim=1),
            "feet_regulation": feet_regulation(
                self._robot.data.body_pos_w[:, self._feet_ids_bd],
                self._robot.data.root_pos_w,
                self._robot.data.body_lin_vel_w[:, self._feet_ids_bd],
                measured_base_height,
                self.cfg.sim.gravity,
                self.cfg.rewards.base_height_target,
            ),
            "hip_pos_penalty_l1": hip_pos_penalty_l1(
                self._robot.data.joint_pos[:, self._hip_ids_jt],
                self._robot.data.default_joint_pos[:, self._hip_ids_jt],
                command,
            ),
            "joint_pos_penalty_l1": joint_pos_penalty_l1(
                self._robot.data.joint_pos[:, self._thigh_calf_ids_jt],
                self._robot.data.default_joint_pos[:, self._thigh_calf_ids_jt],
                command,
                self._robot.data.root_lin_vel_b,
            ),
        }

        reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for name, rate in reward_rates.items():
            weight = self._reward_weights[name]
            if weight == 0.0:
                continue
            value = rate * weight * self.step_dt
            reward_buf += value
            self._episode_sums[name] += value

        return reward_buf

    def _update_reward_weights(self) -> None:
        iteration = self.common_step_counter // 24
        cfg = self.cfg.rewards
        self._reward_weights["lin_vel_z_l2"] = linear_schedule(
            iteration,
            cfg.lin_vel_z_l2_initial,
            cfg.lin_vel_z_l2_final,
            cfg.lin_vel_z_l2_start_iteration,
            cfg.lin_vel_z_l2_end_iteration,
        )
        self._reward_weights["base_height_l2"] = linear_schedule(
            iteration,
            cfg.base_height_l2_initial,
            cfg.base_height_l2_final,
            cfg.base_height_l2_start_iteration,
            cfg.base_height_l2_end_iteration,
        )

    def get_curriculum_state(self) -> dict:
        """Return the environment state needed to continue training curricula."""
        return {
            "common_step_counter": int(self.common_step_counter),
            "terrain_levels": self._terrain.terrain_levels.detach().cpu(),
        }

    def load_curriculum_state(self, state: dict) -> None:
        """Restore time-based and adaptive terrain curricula at a rollout boundary."""
        common_step_counter = int(state["common_step_counter"])
        if common_step_counter < 0:
            raise ValueError("common_step_counter must be non-negative.")
        self.common_step_counter = common_step_counter

        terrain_levels = state.get("terrain_levels")
        if terrain_levels is not None:
            terrain_levels = torch.as_tensor(
                terrain_levels,
                dtype=self._terrain.terrain_levels.dtype,
                device=self.device,
            )
            if terrain_levels.shape == self._terrain.terrain_levels.shape:
                if torch.any((terrain_levels < 0) | (terrain_levels >= self.num_terrain_rows)):
                    raise ValueError("Checkpoint contains invalid terrain curriculum levels.")
                self._terrain.terrain_levels.copy_(terrain_levels)
                self._terrain.env_origins.copy_(
                    self._terrain.terrain_origins[
                        self._terrain.terrain_levels,
                        self._env_terrain_cols,
                    ]
                )
                self._env_goals = self._terrain_goals[
                    self._terrain.terrain_levels,
                    self._env_terrain_cols,
                    :,
                    :,
                ].clone()
                all_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                self._apply_goal_y_shifts(all_envs)
                self._setup_rewards()
            else:
                print(
                    "[WARNING] Not restoring terrain curriculum because the checkpoint has "
                    f"{terrain_levels.numel()} environments but the current run has {self.num_envs}."
                )

        # Reinitialize episodes on the restored terrain.  Suppressing the
        # usual move-up/down update prevents the fresh simulator state from
        # changing the checkpointed terrain levels.  Reset-time schedules
        # (reward weights, gains, and commands) now see the restored counter.
        self._first_reset = True
        self.reset()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        reach_goal_cutoff = (
            self._env_next_goals_idx >= self._terrain.terrain_generator_class.num_goals
        ) & self._waypoint_command_terrain_mask
        time_out |= reach_goal_cutoff

        roll, pitch, _ = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
        roll_cutoff = torch.abs(math_utils.wrap_to_pi(roll)) > 1.5
        pitch_cutoff = torch.abs(math_utils.wrap_to_pi(pitch)) > 1.5
        height_cutoff = self._robot.data.root_state_w[:, 2] < -5.0
        died = roll_cutoff | pitch_cutoff | height_cutoff

        self._reset_buf[:] = died

        self._terminal_obs[:] = self._get_observations(pure=True)["policy"]

        return died, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        env_mask[env_ids] = True

        self._update_reward_weights()

        self._obs_buffer[env_ids, ...] = 0.0

        # update terrain curriculum
        if not self._first_reset:
            # compute curriculum level changes
            num_goals = self._terrain.terrain_generator_class.num_goals
            goals_covered = self._env_next_goals_idx[env_mask]
            move_up = goals_covered >= num_goals
            move_down = goals_covered * 2 < num_goals
            direct_resets = self._direct_command_terrain_mask[env_ids]
            if torch.any(direct_resets):
                direct_env_ids = env_ids[direct_resets]
                terrain_cfg = self.cfg.terrain.terrain_generator
                sub_terrain_border_width = terrain_cfg.sub_terrain_border_width or 0.0
                terrain_length = max(0.0, terrain_cfg.size[0] - 2.0 * sub_terrain_border_width)
                direct_move_up, direct_move_down = direct_command_curriculum_moves(
                    self._max_move_distance[direct_env_ids],
                    self._commands_xy_accumulation[direct_env_ids],
                    terrain_length,
                    self.cfg.cmd_resample_interval[0],
                    self._cmd_not_zero_out_prob[0],
                )
                move_up[direct_resets] = direct_move_up
                move_down[direct_resets] = direct_move_down
            # update env origins and terrain levels
            self._terrain.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
            # robots that solve the last level are sent to a random one
            self._terrain.terrain_levels[env_ids] = torch.where(
                self._terrain.terrain_levels[env_ids] >= self.num_terrain_rows,
                torch.randint_like(self._terrain.terrain_levels[env_ids], self.num_terrain_rows),
                torch.clip(self._terrain.terrain_levels[env_ids], 0),
            )
            if self.cfg.is_play_env:
                self._terrain.terrain_levels[env_ids] = self._ui_terrain_levels[env_ids]
            # update env origins
            self._terrain.env_origins[env_ids, :] = self._terrain.terrain_origins[
                self._terrain.terrain_levels[env_ids], self._env_terrain_cols[env_ids], :
            ]
            self._env_goals[env_ids, :, :] = self._terrain_goals[
                self._terrain.terrain_levels[env_ids], self._env_terrain_cols[env_ids], :, :
            ]
            self._apply_goal_y_shifts(env_mask)
        else:
            self._first_reset = False

        self._commands_xy_accumulation[env_ids] = 0.0
        self._max_move_distance[env_ids] = 0.0

        # Reset the scene, actuator delay, joints, motor offsets, and root
        # state through the ordered RobotLab-compatible reset events.
        super()._reset_idx(env_ids)

        # Gradually shift the gain distribution from (25.0, 0.5) with a
        # [0.9, 1.1] scale to (30.0, 1.0) with a [0.7, 1.3] scale.
        training_iteration = self.common_step_counter // 24
        gain_curriculum_progress = min(max((training_iteration - 5000) / 5000, 0.0), 1.0)
        if self.cfg.is_play_env:
            gain_curriculum_progress = 1.0
        stiffness_center = 25.0 + 5.0 * gain_curriculum_progress
        damping_center = 0.5 + 0.5 * gain_curriculum_progress
        gain_half_range = 0.1 + 0.2 * gain_curriculum_progress
        gain_scale_min = 1.0 - gain_half_range
        gain_scale_width = 2.0 * gain_half_range

        for actuator in self._robot.actuators.values():
            stiffness = actuator.stiffness[env_ids, :]
            damping = actuator.damping[env_ids, :]
            actuator.stiffness[env_ids, :] = stiffness_center * (
                torch.rand_like(stiffness) * gain_scale_width + gain_scale_min
            )
            actuator.damping[env_ids, :] = damping_center * (
                torch.rand_like(damping) * gain_scale_width + gain_scale_min
            )

        # reset actions
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_previous_actions[env_ids] = 0.0

        # reset invalidate prob scale for spherical scan
        self._invalidate_prob_scale[env_ids] = math_utils.sample_uniform(0.0, 2.0, len(env_ids), self.device)

        self._occlusion_level[env_ids] = math_utils.sample_uniform(0.0, 0.4, len(env_ids), device=self.device)
        self._spherical_depth_gen.set_occlusion_level(env_ids, self._occlusion_level[env_ids])

        # reset commands
        self._reset_goals(env_mask)
        self._reset_speed(env_mask)

        # logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Episode/average_speed"] = torch.mean(self._robot.data.root_com_lin_vel_b.norm(dim=1))
        extras["Episode/average_terrain_level"] = torch.mean(self._terrain.terrain_levels.to(torch.float))
        extras["Episode/max_terrain_level"] = torch.max(self._terrain.terrain_levels.to(torch.float))
        for terrain_name in self.cfg.terrain.terrain_generator.sub_terrains.keys():
            mask = (self._env_terrain_names == terrain_name)[:, -1]
            extras[f"Episode/avg_terrain_level_{terrain_name}"] = torch.mean(
                self._terrain.terrain_levels[mask].to(torch.float)
            )
            # extras[f"Episode/count_terrain_{terrain_name}"] = np.count_nonzero(mask)

        max_kp, max_kd = 0.0, 0.0
        min_kp, min_kd = 1000.0, 1000.0
        for _, v in self._robot.actuators.items():
            max_kp = torch.max(v.stiffness[:]).item()
            max_kd = torch.max(v.damping[:]).item()
            min_kp = torch.min(v.stiffness[:]).item()
            min_kd = torch.min(v.damping[:]).item()
        extras["Episode/max_kp"] = max_kp
        extras["Episode/max_kd"] = max_kd
        extras["Episode/min_kp"] = min_kp
        extras["Episode/min_kd"] = min_kd

        self.extras["log"] = extras

    def _reset_goals(self, masks: torch.Tensor | None = None):
        if masks is None:
            masks = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        masks = masks.squeeze()

        num_resets = torch.count_nonzero(masks).item()
        if num_resets == 0:
            return

        self._env_next_goals_idx[masks] = 0
        self._cmd_goal_w[masks, :] = self._env_goals[masks, self._env_next_goals_idx[masks], :]

    def _apply_goal_y_shifts(self, masks: torch.Tensor):
        """Apply random lateral (Y) shifts to waypoints for envs indicated by masks.

        The shift range per waypoint is stored in _terrain_goal_y_shift_ranges and was
        computed by terrain generation (e.g., bounded by stair width for stairs, or by a
        config param for boxes). Flat terrains have zero shift ranges.
        """
        masks = masks.squeeze()
        env_ids = masks.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return

        # Gather per-waypoint shift ranges for the envs being reset
        shift_ranges = self._terrain_goal_y_shift_ranges[
            self._terrain.terrain_levels[env_ids], self._env_terrain_cols[env_ids], :
        ]  # shape: (num_resets, num_goals)

        # Randomly pick one of three discrete positions per waypoint: left (-range), center (0), right (+range)
        choice = torch.randint(0, 3, shift_ranges.shape, device=self.device)  # 0=left, 1=center, 2=right
        multiplier = (choice - 1).float()  # -1, 0, +1
        multiplier *= 0.2
        terrain_names = self._terrain.terrain_generator_class.terrain_names[
            self._terrain.terrain_levels[env_ids].cpu().numpy(), self._env_terrain_cols[env_ids].cpu().numpy()
        ]
        discrete_obstacles = torch.from_numpy(terrain_names[:, -1] == "parkour_discrete_obstacles").to(self.device)
        if torch.any(discrete_obstacles):
            multiplier[discrete_obstacles, :] = torch.randint(
                0, 2, shift_ranges[discrete_obstacles].shape, device=self.device
            ).float()
        random_shifts = multiplier * shift_ranges
        # Apply to Y component of goals
        self._env_goals[env_ids, :, 1] += random_shifts

    def _reset_speed(self, masks: torch.Tensor | None = None):
        # if self.cfg.is_play_env and self.viewport_camera_controller is not None:
        #     return

        if masks is None:
            masks = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        masks = masks.squeeze()

        num_resets = torch.count_nonzero(masks).item()
        if num_resets == 0:
            return

        self._cmd_speed[masks, 0] = (
            math_utils.sample_uniform(0.5, 1.0, num_resets, self.device)
            * (math_utils.sample_uniform(0.0, 1.0, num_resets, self.device) < 0.95).float()
        )

        direct_resets = masks & self._direct_command_terrain_mask
        num_direct_resets = torch.count_nonzero(direct_resets).item()
        if num_direct_resets > 0:
            direct_commands = math_utils.sample_uniform(
                self._cmd_commands_low,
                self._cmd_commands_high,
                (num_direct_resets, 3),
                self.device,
            )
            linear_signs = torch.where(
                torch.rand(num_direct_resets, 2, device=self.device) < 0.5,
                -1.0,
                1.0,
            )
            direct_commands[:, :2] *= linear_signs
            direct_commands *= (
                torch.rand(num_direct_resets, 3, device=self.device) < self._cmd_not_zero_out_prob
            ).float()
            self._cmd_lin_vel[direct_resets, :] = direct_commands[:, :2]
            self._cmd_ang_vel[direct_resets, 0] = direct_commands[:, 2]

        self._cmd_in_place_rotation[masks] = False
        self._cmd_in_place_ang_vel[masks] = 0.0
        if self.common_step_counter // 24 >= 1000:
            in_place_resets = masks & self._direct_command_terrain_mask
            # in_place_resets |= masks & self._stairs_terrain_mask
            num_in_place_resets = torch.count_nonzero(in_place_resets).item()
            if num_in_place_resets > 0:
                self._cmd_in_place_rotation[in_place_resets] = torch.rand(num_in_place_resets, device=self.device) < 0.2
            rotation_resets = masks & self._cmd_in_place_rotation
            num_rotation_resets = torch.count_nonzero(rotation_resets).item()
            if num_rotation_resets > 0:
                self._cmd_in_place_ang_vel[rotation_resets] = math_utils.sample_uniform(
                    -2.0, 2.0, num_rotation_resets, self.device
                )
                self._cmd_lin_vel[rotation_resets, :] = 0.0
                self._cmd_ang_vel[rotation_resets, 0] = self._cmd_in_place_ang_vel[rotation_resets]

        if num_direct_resets > 0:
            self._commands_xy_accumulation[direct_resets] += self._cmd_lin_vel[direct_resets]

        self._cmd_resample_intervals[masks] = math_utils.sample_uniform(
            self.cfg.cmd_resample_interval[0], self.cfg.cmd_resample_interval[1], (num_resets, 1), self.device
        )
        self._cmd_resample_accums[masks] = torch.zeros(num_resets, 1, device=self.device)

        # Omni-directional heading offset: random angle on flat terrain, zero on parkour terrain
        flat_resets = masks & self._flat_terrain_mask.to(masks.device)
        num_flat = torch.count_nonzero(flat_resets).item()
        if num_flat > 0:
            self._cmd_heading_offset[flat_resets] = math_utils.sample_uniform(
                -torch.pi, torch.pi, num_flat, self.device
            )
        non_flat_resets = masks & self._non_flat_terrain_mask.to(masks.device)
        self._cmd_heading_offset[non_flat_resets] = 0.0

        self._cmd_heading_offset[:] = 0.0

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self._goal_vel_viz.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self._robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _get_height_map_points_w(self):
        # height_data = self._height_scanner.data.ray_hits_w[:, :, 2] - self._robot.data.root_pos_w[:, 2:3]
        # height_data = height_data.nan_to_num_(nan=2.0, posinf=2.0, neginf=-2.0)
        # height_data = height_data.clip(min=-2.0, max=2.0) * 0.1

        # num_rays = self._height_scanner.ray_directions.shape[1]
        # ray_starts_w = (
        #     math_utils.quat_apply_yaw(
        #         self._robot.data.root_quat_w[:, None, :].repeat(1, num_rays, 1),
        #         self._height_scanner.ray_starts,
        #     )
        #     + self._robot.data.root_pos_w[:, None, :]
        # )
        # ray_dirs_w = math_utils.quat_apply_yaw(
        #     self._robot.data.root_quat_w[:, None, :].repeat(1, num_rays, 1),
        #     self._height_scanner.ray_directions,
        # )
        # points_w = ray_starts_w + ray_dirs_w * height_data.unsqueeze(-1) * 10.0

        points_w = self._height_scanner.data.ray_hits_w.clone()
        return points_w

    def _update_debug_draw(self):
        if not self._show_debug_viz:
            return

        self._debug_draw.clear_lines()
        self._debug_draw.clear_points()

        # visualize commands
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._cmd_lin_vel[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self._robot.data.root_lin_vel_b[:, :2])
        # display markers
        self._goal_vel_viz.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self._cur_vel_viz.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

        # # visualize raycaster position
        # raycaster_offset = torch.tensor(self.cfg.mid360.offset.pos, device=self.device)
        # raycaster_offset_w = math_utils.quat_apply(self._robot.data.root_quat_w[self._track_env_id], raycaster_offset)
        # raycaster_pos = (self._robot.data.root_pos_w[self._track_env_id] + raycaster_offset_w).unsqueeze(0)
        # self._mid360_viz.visualize(raycaster_pos)

        # # visualize mid360 pointcloud
        # self._pointcloud_viz.visualize(self._mid360.data.ray_hits_w[self._track_env_id])

        # # visualize GT spherical scan
        # _, points_w = self._get_spherical_scan()
        # self._spherical_scan_viz.visualize(points_w[self._track_env_id])

        # visualize height map points
        height_map_points_w = self._get_height_map_points_w()
        self._height_map_viz.visualize(height_map_points_w[self._track_env_id])

        # # visualize projected spherical grid from mid360
        # _, projected_points_w = self._project_mid360_points_to_spherical_scan()
        # self._projected_spherical_scan_viz.visualize(projected_points_w[self._track_env_id])

        # --- visualize env goals ---
        goals = self._env_goals[self._track_env_id]
        self._env_goals_viz.visualize(goals)
        self._env_next_goals_viz.visualize(self._cmd_goal_w[self._track_env_id].unsqueeze(0))
