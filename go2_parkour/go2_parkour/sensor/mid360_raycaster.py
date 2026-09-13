# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab ray caster with the time-varying Livox Mid360 scan pattern."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply

from .livox_scan_generator import LivoxScanGenerator


class Mid360RayCaster(RayCaster):
    """Built-in Isaac Lab ray caster driven by the Livox Mid360 scan pattern."""

    def _initialize_rays_impl(self):
        pattern_cfg = self.cfg.pattern_cfg
        self._scan_generator = LivoxScanGenerator(
            name="mid360",
            num_envs=self._view.count,
            device=self._device,
            horizontal_fov_deg=pattern_cfg.horizontal_fov_deg,
            vertical_fov_deg=pattern_cfg.vertical_fov_deg,
        )
        self._offset_pos = torch.tensor(self.cfg.offset.pos, device=self._device)
        self._offset_quat = torch.tensor(self.cfg.offset.rot, device=self._device)

        self.ray_starts, self.ray_directions = self._scan_generator.sample_rays()
        self.num_rays = self.ray_directions.shape[1]
        self._apply_offset(self.ray_starts, self.ray_directions)

        self.drift = torch.zeros(self._view.count, 3, device=self.device)
        self.ray_cast_drift = torch.zeros(self._view.count, 3, device=self.device)
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        self._ray_starts_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        self._ray_directions_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)

        print(f"Mid360 rays per frame: {self.num_rays}")

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        ray_starts, ray_directions = self._scan_generator.sample_rays(env_ids=env_ids)
        self._apply_offset(ray_starts, ray_directions)
        self.ray_starts[env_ids] = ray_starts
        self.ray_directions[env_ids] = ray_directions
        super()._update_buffers_impl(env_ids)

    def _apply_offset(self, ray_starts: torch.Tensor, ray_directions: torch.Tensor):
        num_envs, num_rays = ray_directions.shape[:2]
        ray_starts += self._offset_pos
        ray_directions[:] = quat_apply(
            self._offset_quat.repeat(num_envs, num_rays, 1),
            ray_directions,
        )
