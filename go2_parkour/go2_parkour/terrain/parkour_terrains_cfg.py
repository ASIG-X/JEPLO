from go2_parkour.terrain.parkour_sub_terrains_cfg import *  # noqa: F403
from go2_parkour.terrain.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg

PARKOUR_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(9.0, 9.0),
    border_width=20.0,
    sub_terrain_border_width=0.5,
    num_rows=10,
    num_cols=30,
    horizontal_scale=0.08,  ## original scale is 0.05, But Computing issue in IsaacLab see this issue in https://github.com/isaac-sim/IsaacLab/issues/2187
    vertical_scale=0.005,
    slope_threshold=0.5,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "parkour_gap": ExtremeParkourGapTerrainCfg(
            proportion=0.25,
            apply_roughness=True,
            max_num_gaps=6,
            platform_len=1.0,
            x_range=(0.8, 1.0),
            half_valid_width=(0.6, 1.2),
            gap_size="0.1 + 0.54*difficulty",
            gap_depth_start=(1.0, 2.0),
            gap_depth=(1.0, 2.0),
            gap_depth_difficulty_power=2.0,
        ),
        # "parkour_hurdle": ExtremeParkourHurdleTerrainCfg(
        #     proportion=0.2,
        #     apply_roughness=True,
        #     x_range=(1.2, 2.2),
        #     half_valid_width=(0.4, 0.8),
        #     hurdle_height_range="0.1+0.1*difficulty, 0.15+0.25*difficulty",
        # ),
        "parkour_flat": ExtremeParkourHurdleTerrainCfg(
            proportion=0.25,
            apply_roughness=True,
            apply_flat=True,
            x_range=(1.2, 2.2),
            half_valid_width=(0.4, 0.8),
            hurdle_height_range="0.1+0.1*difficulty, 0.15+0.15*difficulty",
        ),
        # "parkour_stairs": ExtremeParkourStairsTerrainCfg(
        #     proportion=0.25,
        #     apply_roughness=True,  # flat stairs
        #     step_height="0.04 + 0.16*difficulty",  # 0.04m to 0.2m per step
        #     step_depth_range=(0.3, 0.4),  # randomized step depth per subterrain
        #     num_steps=6,  # steps per staircase (up or down)
        #     num_stair_sets=1,  # one centered set of up-down stairs
        #     stair_width_range=(1.5, 3.0),  # randomized width (perpendicular to movement)
        #     flat_spacing=1.0,  # flat ground between stair sets
        #     noise_range=(0.01, 0.03),
        # ),
        "parkour_box": ExtremeParkourBoxTerrainCfg(
            proportion=0.25,
            apply_roughness=True,
            box_length_range=(0.8, 1.5),  # length along robot movement (X direction)
            box_height="0.1 + 0.4*difficulty",  # 0.1m to 0.55m height
            noise_range=(0.01, 0.03),
        ),
        # "parkour_pyramid_slope": ExtremeParkourPyramidSlopeTerrainCfg(
        #     proportion=0.25,
        #     apply_roughness=False,
        #     slope_range=(0.0, 0.2),
        #     platform_width=1.5,
        #     zigzag_y_range=(0.5, 1.0),
        #     zigzag_y_margin=0.3,
        # ),
        # "parkour_inverted_pyramid_slope": ExtremeParkourInvertedPyramidSlopeTerrainCfg(
        #     proportion=0.25,
        #     apply_roughness=False,
        #     slope_range=(0.0, 0.2),
        #     platform_width=1.5,
        #     zigzag_y_range=(0.5, 1.0),
        #     zigzag_y_margin=0.3,
        # ),
        # "parkour_wave": ExtremeParkourWaveTerrainCfg(
        #     proportion=0.25,
        #     apply_roughness=False,
        #     amplitude_range=(0.02, 0.12),
        #     num_waves_range=(3.0, 6.0),
        #     zigzag_y_range=(0.5, 1.0),
        #     zigzag_y_margin=0.3,
        # ),
        "parkour_discrete_obstacles": ExtremeParkourDiscreteObstaclesTerrainCfg(
            proportion=0.25,
            apply_roughness=True,
            noise_range=(0.01, 0.03),
            grid_cell_size=(0.5, 0.5),
            grid_height_range=(0.0, 0.1),
            box_height_range=(0.0, 0.4),
            box_size_range=(0.8, 1.25),
            box_rotation_range=(0.0, 90.0),
            goal_clearance=(0.35, 0.5),
            goal_y_shift_range=0.8,
            zigzag_y_range=(0.8, 1.2),
            height_difficulty_power=1.0,
            num_obstacle_rows=3,
            obstacle_row_spacing=2.0,
        ),
        "parkour_discrete_grid": ExtremeParkourDiscreteGridTerrainCfg(
            proportion=0.25,
            apply_roughness=True,
            noise_range=(0.01, 0.03),
            grid_cell_size=(0.5, 0.5),
            grid_height_range=(0.0, 0.2),
            height_difficulty_power=1.0,
        ),
        # "parkour_step": ExtremeParkourStepTerrainCfg(
        #     proportion=0.2,
        #     apply_roughness=True,
        #     x_range=(0.3, 1.5),
        #     half_valid_width=(0.5, 1),
        #     step_height="0.1 + 0.35*difficulty",
        # ),
        # "parkour": ExtremeParkourTerrainCfg(
        #     proportion=0.2,
        #     apply_roughness=True,
        #     x_range="-0.1, 0.1+0.3*difficulty",
        #     y_range="0.2, 0.3+0.1*difficulty",
        #     stone_len="0.9 - 0.3*difficulty, 1 - 0.2*difficulty",
        #     incline_height="0.25*difficulty",
        #     last_incline_height="incline_height + 0.1 - 0.1*difficulty",
        # ),
        # "parkour_stairs": ExtremeParkourStairsTerrainCfg(
        #     proportion=0.1,
        #     apply_roughness=True,
        #     num_steps=5,
        # ),
        # "parkour_demo": ExtremeParkourDemoTerrainCfg(
        #     proportion=0.0,
        #     apply_roughness=True,
        # ),
        "parkour_pyramid_stairs": ExtremeParkourPyramidStairsTerrainCfg(
            proportion=0.25,
            apply_roughness=True,
            step_height_range=(0.04, 0.2),
            step_width_range=(0.25, 0.35),
            platform_width=2.0,
            pyramid_length=12.0,
            goal_y_shift_range=2.0,
            noise_range=(0.01, 0.02),
        ),
        "parkour_inverted_pyramid_stairs": ExtremeParkourInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            apply_roughness=True,
            step_height_range=(0.04, 0.2),
            step_width_range=(0.25, 0.35),
            platform_width=2.0,
            pyramid_length=12.0,
            goal_y_shift_range=2.0,
            noise_range=(0.01, 0.02),
        ),
    },
)
