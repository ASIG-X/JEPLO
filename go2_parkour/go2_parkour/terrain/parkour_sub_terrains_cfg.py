from go2_parkour.terrain import parkour_sub_terrians as parkour_terrians
from go2_parkour.terrain.parkour_terrain_generator_cfg import ParkourSubTerrainBaseCfg
from isaaclab.utils import configclass


@configclass
class ExtremeParkourRoughTerrainCfg(ParkourSubTerrainBaseCfg):
    apply_roughness: bool = True
    apply_flat: bool = False
    downsampled_scale: float | None = 0.075
    noise_range: tuple[float, float] = (0.02, 0.06)
    noise_step: float = 0.005
    x_range: tuple[float, float] = (0.8, 1.5)
    y_range: tuple[float, float] = (-0.4, 0.4)
    half_valid_width: tuple[float, float] = (0.6, 1.2)
    pad_width: float = 0.1
    pad_height: float = 0.0


@configclass
class ExtremeParkourGapTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = parkour_terrians.parkour_gap_terrain
    max_num_gaps: int = 6
    gap_size: str = "0.1 + 0.54*difficulty"
    gap_depth_start: tuple[float, float] = (0.05, 0.08)
    gap_depth: tuple[float, float] = (0.2, 1)
    gap_depth_difficulty_power: float = 2.0


@configclass
class ExtremeParkourHurdleTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = parkour_terrians.parkour_hurdle_terrain
    stone_len: str = "0.1 + 0.3 * difficulty"
    hurdle_height_range: str = "0.1 + 0.1 * difficulty, 0.15 + 0.15 * difficulty"


@configclass
class ExtremeParkourStepTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = parkour_terrians.parkour_step_terrain
    step_height: str = "0.1 + 0.35*difficulty"


@configclass
class ExtremeParkourTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = parkour_terrians.parkour_terrain
    pit_depth: tuple[float, float] = (0.2, 1)
    stone_width: float = 1.0
    last_stone_len: float = 1.6
    x_range: str = "-0.1, 0.1+0.3*difficulty"
    y_range: str = "0.2, 0.3+0.1*difficulty"
    stone_len: str = "0.9 - 0.3*difficulty, 1 - 0.2*difficulty"
    incline_height: str = "0.25*difficulty"
    last_incline_height: str = "incline_height + 0.1 - 0.1*difficulty"


@configclass
class ExtremeParkourStairsTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """Uniform stairs terrain with configurable centered up-down stair sets."""

    function = parkour_terrians.parkour_stairs_terrain
    step_height: str = "0.08 + 0.12*difficulty"  # height of each step (0.08m to 0.2m)
    step_depth_range: tuple[float, float] = (0.25, 0.4)  # randomized depth of each step
    num_steps: int = 4  # number of steps per staircase (up or down)
    num_stair_sets: int = 2  # number of up-down stair sets
    stair_width_range: tuple[float, float] = (1.5, 3.0)  # randomized width of staircase (perpendicular to movement)
    flat_spacing: float = 1.0  # flat ground spacing between stair sets


@configclass
class ExtremeParkourBoxTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """Box terrain - robot climbs over step-like platforms."""

    function = parkour_terrians.parkour_box_terrain
    box_length_range: tuple[float, float] = (1.0, 2.0)  # length of boxes along robot movement (X direction)
    box_height: str = "0.1 + 0.3*difficulty"  # height of boxes (0.1m to 0.4m)


@configclass
class ExtremeParkourPyramidSlopeTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """Pyramid slope terrain - IsaacLab-style sloped surface with a center plateau."""

    function = parkour_terrians.parkour_pyramid_slope_terrain
    slope_range: tuple[float, float] = (0.0, 0.12)
    platform_width: float = 1.5
    inverted: bool = False
    disable_slope_correction: bool = True
    zigzag_y_range: tuple[float, float] | None = (0.5, 1.0)
    zigzag_y_margin: float = 0.3


@configclass
class ExtremeParkourInvertedPyramidSlopeTerrainCfg(ExtremeParkourPyramidSlopeTerrainCfg):
    """Inverted pyramid slope terrain - center plateau is below the terrain edges."""

    inverted: bool = True


@configclass
class ExtremeParkourWaveTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """Wave terrain - IsaacLab-style sinusoidal height field."""

    function = parkour_terrians.parkour_wave_terrain
    amplitude_range: tuple[float, float] = (0.02, 0.12)
    num_waves: float = 3.0
    num_waves_range: tuple[float, float] | None = (3.0, 6.0)
    disable_slope_correction: bool = True
    zigzag_y_range: tuple[float, float] | None = (0.5, 1.0)
    zigzag_y_margin: float = 0.3


@configclass
class ExtremeParkourPyramidStairsTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """Centered elongated pyramid stairs with zigzag waypoints."""

    function = parkour_terrians.parkour_pyramid_stairs_terrain
    step_height_range: tuple[float, float] = (0.05, 0.23)
    step_width: float = 0.3
    step_width_range: tuple[float, float] | None = None
    platform_width: float = 1.5
    pyramid_length: float = 12.0
    inverted: bool = False
    goal_y_shift_range: float = 2.0


@configclass
class ExtremeParkourInvertedPyramidStairsTerrainCfg(ExtremeParkourPyramidStairsTerrainCfg):
    """Centered elongated inverted pyramid stairs with zigzag waypoints."""

    inverted: bool = True


@configclass
class ExtremeParkourDiscreteObstaclesTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """Rotated boxes under waypoint candidates."""

    function = parkour_terrians.parkour_discrete_obstacles_terrain
    add_boxes: bool = True
    grid_cell_size: tuple[float, float] = (1.0, 1.0)
    grid_height_range: tuple[float, float] = (0.0, 0.1)
    box_height_range: tuple[float, float] = (0.0, 0.25)
    box_size_range: tuple[float, float] = (1.0, 2.0)
    box_rotation_range: tuple[float, float] = (0.0, 90.0)
    goal_clearance: tuple[float, float] = (0.35, 0.5)
    goal_y_shift_range: float = 0.8
    zigzag_y_range: tuple[float, float] = (0.6, 1.2)
    height_difficulty_power: float = 2.0
    num_obstacle_rows: int = 3
    obstacle_row_spacing: float = 2.0


@configclass
class ExtremeParkourDiscreteGridTerrainCfg(ExtremeParkourDiscreteObstaclesTerrainCfg):
    """Discrete height grid without boxes."""

    add_boxes: bool = False


@configclass
class ExtremeParkourDemoTerrainCfg(ExtremeParkourRoughTerrainCfg):
    function = parkour_terrians.parkour_demo_terrain
