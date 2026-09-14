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

from typing import TYPE_CHECKING

import numpy as np
from isaaclab.envs.ui import BaseEnvWindow

if TYPE_CHECKING:
    import omni.ui as ui

    from .go2_loco_env import Go2LocoEnv


class Go2LocoEnvWindow(BaseEnvWindow):
    """IsaacLab environment window extended with Go2 parkour playback controls."""

    _THIGH_JOINTS = (
        ("Front left", "FL_thigh_joint"),
        ("Front right", "FR_thigh_joint"),
        ("Rear left", "RL_thigh_joint"),
        ("Rear right", "RR_thigh_joint"),
    )
    _WINDOWS_TO_CLOSE = (
        "Content",
        "Console",
        "IsaacLab",
        "Semantics Schema Editor",
        "Property",
    )

    def __init__(self, env: Go2LocoEnv, window_name: str = "IsaacLab"):
        import omni.ui as ui

        del window_name
        self._ui = ui
        env.initialize_playback_controls()
        self._thigh_joint_ids = [env._robot.joint_names.index(name) for _, name in self._THIGH_JOINTS]
        super().__init__(env, window_name="Simulation Controls")

        self.ui_window_elements["sim_frame"].collapsed = True
        self.ui_window_elements["viewer_frame"].collapsed = True
        self.ui_window_elements["debug_frame"].collapsed = True
        self._create_image_providers()
        self._build_parkour_controls()
        self.update_thigh_joint_angles()

    def _create_image_providers(self) -> None:
        ui = self._ui
        depth_pattern = self.env.cfg.mid360.pattern_cfg
        self._depth_height = depth_pattern.vertical_num_rays
        self._depth_width = depth_pattern.horizontal_num_rays
        self._depth_image_provider = ui.ByteImageProvider()
        self._set_blank_image(self._depth_image_provider, self._depth_height, self._depth_width)

        height_pattern = self.env.cfg.height_scanner.pattern_cfg
        self._height_map_cols = round(height_pattern.size[0] / height_pattern.resolution)
        self._height_map_rows = round(height_pattern.size[1] / height_pattern.resolution)
        self._height_map_provider = ui.ByteImageProvider()
        self._set_blank_image(
            self._height_map_provider,
            self._height_map_rows,
            self._height_map_cols,
        )

    @staticmethod
    def _set_blank_image(provider: ui.ByteImageProvider, rows: int, cols: int) -> None:
        image = np.zeros((rows, cols, 4), dtype=np.uint8)
        image[..., 3] = 255
        provider.set_bytes_data(image.flatten().tolist(), [cols, rows])

    def _build_parkour_controls(self) -> None:
        ui = self._ui
        with self.ui_window_elements["main_vstack"]:
            self.ui_window_elements["parkour_frame"] = ui.CollapsableFrame(
                title="Parkour Controls",
                width=ui.Fraction(1),
                height=0,
                collapsed=False,
            )
            with self.ui_window_elements["parkour_frame"]:
                with ui.VStack(spacing=5, height=0):
                    ui.Label("Switching Tracked Env", style={"color": 0xFFFFFFFF, "font_size": 16})
                    ui.Button("Next Env", clicked_fn=lambda: self.env.cycle_tracked_env(1))
                    ui.Button("Prev Env", clicked_fn=lambda: self.env.cycle_tracked_env(-1))
                    ui.Button("Reset Env", clicked_fn=self.env.reset_tracked_env)
                    ui.Button("Next Terrain Level", clicked_fn=lambda: self.env.change_tracked_terrain_level(1))
                    ui.Button("Prev Terrain Level", clicked_fn=lambda: self.env.change_tracked_terrain_level(-1))
                    ui.Button("Toggle In-Place Rotation", clicked_fn=self.env.toggle_tracked_in_place_rotation)
                    ui.Button("Toggle Track Camera", clicked_fn=self.env.toggle_camera_tracking)
                    ui.Label("Thigh Joint Angles", style={"color": 0xFFFFFFFF, "font_size": 16})
                    self._thigh_env_label = ui.Label(
                        "Tracked env -- | degrees with radians in parentheses",
                        style={"color": 0xFFAAAAAA},
                    )
                    with ui.HStack(spacing=8, height=22):
                        ui.Label("Leg", width=90, style={"color": 0xFFFFFFFF})
                        ui.Label("Current", width=135, style={"color": 0xFF77DDFF})
                        ui.Label("Default", width=135, style={"color": 0xFFCCCCCC})
                    self._thigh_angle_labels = []
                    for leg_name, _ in self._THIGH_JOINTS:
                        with ui.HStack(spacing=8, height=22):
                            ui.Label(leg_name, width=90)
                            current_label = ui.Label("--", width=135, style={"color": 0xFF77DDFF})
                            default_label = ui.Label("--", width=135, style={"color": 0xFFCCCCCC})
                        self._thigh_angle_labels.append((current_label, default_label))
                    ui.Label("Depth Buffer", style={"color": 0xFFFFFFFF, "font_size": 16})
                    with ui.Frame(width=self._depth_width * 4.0, height=self._depth_height * 4.0):
                        ui.ImageWithProvider(self._depth_image_provider)
                    ui.Label("GT Height Map", style={"color": 0xFFFFFFFF, "font_size": 16})
                    with ui.Frame(width=self._height_map_cols * 8.0, height=self._height_map_rows * 8.0):
                        ui.ImageWithProvider(self._height_map_provider)

    def update_sensor_images(self, height_data) -> None:
        """Refresh the depth and height-map panels for the tracked environment."""

        env_id = self.env.tracked_env_id
        if self.env.common_step_counter % self.env.cfg.depth_update_freq == 0:
            self.update_thigh_joint_angles()
            depth = self.env._depth_buffer[env_id, -1 - self.env.cfg.depth_delay].detach().cpu().numpy()
            depth_grayscale = (np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8)
            self._depth_image_provider.set_bytes_data(
                self._to_rgba(depth_grayscale).flatten().tolist(),
                [self._depth_width, self._depth_height],
            )

        height_map = height_data[env_id].detach().cpu().numpy().reshape(
            self._height_map_rows,
            self._height_map_cols,
        )
        height_min = float(height_map.min())
        height_range = float(height_map.max()) - height_min
        if height_range > 1.0e-6:
            height_grayscale = ((height_map - height_min) / height_range * 255.0).astype(np.uint8)
        else:
            height_grayscale = np.full(height_map.shape, 128, dtype=np.uint8)
        self._height_map_provider.set_bytes_data(
            self._to_rgba(height_grayscale).flatten().tolist(),
            [self._height_map_cols, self._height_map_rows],
        )

    def update_thigh_joint_angles(self) -> None:
        """Refresh current and default thigh angles for the tracked environment."""

        env_id = self.env.tracked_env_id
        self._thigh_env_label.text = f"Tracked env {env_id} | degrees with radians in parentheses"
        current_angles = (
            self.env._robot.data.joint_pos[env_id, self._thigh_joint_ids].detach().cpu().numpy()
        )
        default_angles = (
            self.env._robot.data.default_joint_pos[env_id, self._thigh_joint_ids].detach().cpu().numpy()
        )

        for current, default, labels in zip(current_angles, default_angles, self._thigh_angle_labels):
            labels[0].text = self._format_angle(current)
            labels[1].text = self._format_angle(default)

    @staticmethod
    def _format_angle(angle_radians: float) -> str:
        """Format an angle for quick comparison by a human operator."""

        angle_degrees = np.degrees(angle_radians)
        return f"{angle_degrees:+6.1f}° ({angle_radians:+.3f})"

    @staticmethod
    def _to_rgba(grayscale: np.ndarray) -> np.ndarray:
        rgba = np.empty((*grayscale.shape, 4), dtype=np.uint8)
        rgba[..., :3] = grayscale[..., None]
        rgba[..., 3] = 255
        return rgba

    async def _dock_window(self, window_title: str) -> None:
        """Dock the Go2 controls with Stage and hide unrelated workspace windows."""

        ui = self._ui
        control_window = None
        stage_window = None
        docked = False
        for _ in range(20):
            await self.env.sim.app.next_update_async()
            control_window = ui.Workspace.get_window(window_title)
            stage_window = ui.Workspace.get_window("Stage")

            for title in self._WINDOWS_TO_CLOSE:
                window = ui.Workspace.get_window(title)
                if window is not None and window is not control_window:
                    window.visible = False

            if not docked and control_window is not None and stage_window is not None:
                control_window.dock_in(stage_window, ui.DockPosition.SAME)
                docked = True

        if control_window is not None:
            control_window.focus()
