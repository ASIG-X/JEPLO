# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from go2_parkour.assets.unitree import GO2_CFG_UNITREE
from go2_parkour.sensor.lidar_pattern import RaySphericalSlicePatternCfg
from go2_parkour.tasks.go2_loco_events import (
    randomize_motor_zero_offset as randomize_motor_zero_offset_event,
    reset_root_state_uniform as reset_root_state_uniform_event,
)
from go2_parkour.terrain.parkour_terrain_importer import ParkourTerrainImporter
from go2_parkour.terrain.parkour_terrains_cfg import PARKOUR_TERRAINS_CFG


@configclass
class EventCfg:
    """Configuration for events."""

    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-2.0, 5.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!.*base).*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    randomize_motor_zero_offset = EventTerm(
        func=randomize_motor_zero_offset_event,
        mode="reset",
        params={"offset_range": (-0.035, 0.035)},
    )
    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "velocity_range": {
                "x": (-0.4, 0.4),
                "y": (-0.4, 0.4),
                "roll": (-0.6, 0.6),
                "pitch": (-0.6, 0.6),
                "yaw": (-0.6, 0.6),
            }
        },
    )
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 2.0),
            "dynamic_friction_range": (0.0, 2.0),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    reset_base = EventTerm(
        func=reset_root_state_uniform_event,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


@configclass
class RewardCfg:
    """RobotLab Go2 reward weights and schedules."""

    base_height_target = 0.38
    track_lin_vel_xy_exp = 2.0
    track_ang_vel_z_exp = 1.0
    lin_vel_z_l2_initial = -2.0
    lin_vel_z_l2_final = 0.0
    lin_vel_z_l2_start_iteration = 0
    lin_vel_z_l2_end_iteration = 1500
    ang_vel_xy_l2 = -0.05
    joint_acc_l2 = -1.0e-7
    joint_power = -2.0e-5
    joint_torques_l2 = -1.0e-4
    base_height_l2_initial = -1.0
    base_height_l2_final = -10.0
    base_height_l2_start_iteration = 0
    base_height_l2_end_iteration = 5000
    action_rate_l2 = -0.01
    action_smoothness_l2 = -0.01
    undesired_contacts = -1.0
    undesired_contact_threshold = 5.0
    joint_pos_limits = -2.0
    feet_regulation = -0.05
    hip_pos_penalty_l1 = -0.05
    joint_pos_penalty_l1 = -0.01


@configclass
class Go2LocoEnvCfg(DirectRLEnvCfg):
    ui_window_class_type: type | None = None
    episode_length_s = 25.0
    decimation = 4

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**18),
    )

    state_space = 0

    action_space = 12
    action_scale = 0.25
    action_target_clip = (-100.0, 100.0)
    is_play_env = False

    cmd_resample_interval = (4.0, 4.0)

    scene = InteractiveSceneCfg(num_envs=4096, env_spacing=10, replicate_physics=True)
    events: EventCfg = EventCfg()
    rewards: RewardCfg = RewardCfg()

    # ground terrain
    terrain = TerrainImporterCfg(
        class_type=ParkourTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PARKOUR_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=None,
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = GO2_CFG_UNITREE.replace(prim_path="/World/envs/env_.*/Robot")

    # height map raycaster
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 10.0)),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0 - 2e-9, 1.5 - 2e-9)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
    )

    # Reward-only local ground-height scanner used by the RobotLab reward definitions.
    height_scanner_small = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.4, 0.3)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        ray_alignment="yaw",
    )

    # mid360 lidar
    mid360 = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.275, 0.0, 0.148), rot=(0.0, 1.0, 0.0, 0.0)),
        drift_range=(-0.03, 0.03),
        ray_alignment="base",
        pattern_cfg=RaySphericalSlicePatternCfg(),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=2.0,
    )

    # lidar spherical depth image
    lidar_num_stacked: int = 10
    lidar_noise_std: float = 0.005
    lidar_bottom_dropout_fraction: float = 0.05

    num_obs_hist = 10
    num_depth_stack = 2
    depth_delay = 1
    depth_update_freq = 5
    obs_proprio_dim = 45
    obs_proprio_hist_dim = obs_proprio_dim * num_obs_hist
    obs_height_scan_dim = 300
    obs_priv_dim = 50
    obs_critic_proprio_dim = obs_proprio_dim

    obs_proprio_hist_range = (0, obs_proprio_hist_dim)
    obs_height_scan_range = (obs_proprio_hist_range[1], obs_proprio_hist_range[1] + obs_height_scan_dim)
    obs_priv_range = (obs_height_scan_range[1], obs_height_scan_range[1] + obs_priv_dim)
    obs_critic_proprio_range = (obs_priv_range[1], obs_priv_range[1] + obs_critic_proprio_dim)
    observation_space = obs_critic_proprio_range[1]

    obs_curr_proprio_range = (obs_proprio_hist_range[0], obs_proprio_hist_range[0] + obs_proprio_dim)
    obs_curr_proprio_actions_range = (obs_curr_proprio_range[0] + 33, obs_curr_proprio_range[0] + 45)
    obs_curr_proprio_feet_contact_range = (
        obs_curr_proprio_actions_range[1],
        obs_curr_proprio_actions_range[1] + 4,
    )
    obs_curr_proprio_terrain_flat_one_hot_range = (
        obs_curr_proprio_feet_contact_range[1],
        obs_curr_proprio_feet_contact_range[1] + 2,
    )
    obs_curr_proprio_cmd_range = (obs_curr_proprio_range[0] + 6, obs_curr_proprio_range[0] + 9)
    obs_base_lin_vel_dim = 3
    obs_base_lin_vel_range = (obs_priv_range[0], obs_priv_range[0] + obs_base_lin_vel_dim)
    obs_foot_clearance_dim = 4
    obs_foot_clearance_range = (obs_base_lin_vel_range[1], obs_base_lin_vel_range[1] + obs_foot_clearance_dim)

    # sensors
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
        track_pose=True,
    )

    def __post_init__(self):
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_external_forces_every_iteration = True
        # self.sim.physx.gpu_max_rigid_patch_count = int(1 * 1024 * 1024)  # 1 million
        self.sim.physx.gpu_collision_stack_size = int(512 * 1024 * 1024)  # 128 MB

        self.contact_sensor.update_period = self.sim.dt
        self.height_scanner.update_period = self.sim.dt * self.decimation
        self.height_scanner_small.update_period = self.sim.dt * self.decimation
        self.mid360.update_period = self.sim.dt * self.decimation

        sub_terrains = self.terrain.terrain_generator.sub_terrains
        uniform_proportion = 1.0 / len(sub_terrains)
        for sub_terrain in sub_terrains.values():
            sub_terrain.proportion = uniform_proportion
