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

"""Spherical depth image generator from accumulated Mid360 LiDAR frames."""

from __future__ import annotations

import math

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation

from go2_parkour.utils.depth_noise import perlin_2d_batch


class SphericalDepthGenerator:
    """Accumulates temporal LiDAR frames and projects them into a spherical depth image.

    Each simulation step, the raw Mid360 ray hits are projected into a spherical grid
    (V x H bins) via scatter_reduce. Multiple frames are accumulated and aggregated
    (min over time) to produce a dense depth image that can be used in place of a
    depth camera.

    Grid parameters (vertical/horizontal FOV, num rays) and max_distance are read
    directly from ``scanner.cfg.pattern_cfg`` and ``scanner.cfg.max_distance``.

    Args:
        scanner: The Mid360 ray-caster sensor.
        robot: The robot articulation (for computing sensor world pose).
        num_envs: Number of parallel environments.
        device: Torch device string.
        num_stacked: Number of historical frames to aggregate.
        update_freq: Aggregate every N env steps.
        noise_std: Gaussian noise standard deviation (metres).
        bottom_dropout_fraction: Fraction of bottom rows subject to dropout.
    """

    def __init__(
        self,
        scanner,
        robot: Articulation,
        num_envs: int,
        device: str,
        num_stacked: int = 5,
        update_freq: int = 5,
        noise_std: float = 0.005,
        bottom_dropout_fraction: float = 0.05,
        occlusion_level: torch.Tensor | None = None,
        occlusion_perlin_scale: float = 8.0,
        occlusion_perlin_time_scale: float = 0.01,
    ):
        self.scanner = scanner
        self.robot = robot
        self.num_envs = num_envs
        self.device = device

        # Read grid parameters from the scanner's pattern config
        pcfg = scanner.cfg.pattern_cfg
        self.vertical_fov_deg = pcfg.vertical_fov_deg
        self.horizontal_fov_deg = pcfg.horizontal_fov_deg
        self.vertical_num_rays = pcfg.vertical_num_rays
        self.horizontal_num_rays = pcfg.horizontal_num_rays
        self.max_distance = scanner.cfg.max_distance

        self.num_stacked = num_stacked
        self.update_freq = update_freq
        self.noise_std = noise_std
        self.bottom_dropout_fraction = bottom_dropout_fraction

        # Precompute angular parameters
        self._theta_low = math.radians(self.vertical_fov_deg[0])
        self._theta_high = math.radians(self.vertical_fov_deg[1])
        self._phi_low = math.radians(self.horizontal_fov_deg[0])
        self._phi_high = math.radians(self.horizontal_fov_deg[1])
        self._theta_res = (self._theta_high - self._theta_low) / self.vertical_num_rays
        self._phi_res = (self._phi_high - self._phi_low) / self.horizontal_num_rays
        self._phi_res_deg = (self.horizontal_fov_deg[1] - self.horizontal_fov_deg[0]) / self.horizontal_num_rays

        # Precompute sensor offset tensors
        self._offset_pos = torch.tensor(list(scanner.cfg.offset.pos), device=device)
        self._offset_rot = torch.tensor(list(scanner.cfg.offset.rot), device=device)

        # Buffers
        V, H = self.vertical_num_rays, self.horizontal_num_rays
        self.hist_scans = torch.ones(num_envs, num_stacked, V, H, device=device)
        self._aggregated = torch.ones(num_envs, V, H, device=device)

        # Random shape occlusion mask (0, 1 or 2 shapes per env, resampled each output frame).
        # Shape types: 0=sphere, 1=rectangle, 2=square, 3=trapezoid.
        # Pre-built coordinate grid used for rasterization.
        rows = torch.arange(V, device=device, dtype=torch.float32)
        cols = torch.arange(H, device=device, dtype=torch.float32)
        self._grid_r, self._grid_c = torch.meshgrid(rows, cols, indexing="ij")  # (V, H)
        self._cage_mask = torch.zeros(num_envs, V, H, dtype=torch.bool, device=device)
        # self._build_cage_mask(torch.arange(num_envs, device=device))

        # Occlusion level: per-env Perlin-noise-based information loss
        if occlusion_level is not None:
            self._occlusion_level = occlusion_level.clone()
        else:
            self._occlusion_level = torch.zeros(num_envs, device=device)
        self._occlusion_perlin_scale = occlusion_perlin_scale
        self._occlusion_perlin_time_scale = occlusion_perlin_time_scale
        P = 256
        self._occlusion_perm = self._sample_batched_permutations(num_envs, P)
        self._occlusion_perlin_z = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset buffers for given environments."""
        self.hist_scans[env_ids] = 1.0
        self._aggregated[env_ids] = 1.0

    def set_occlusion_level(self, env_ids: torch.Tensor, levels: torch.Tensor) -> None:
        """Update per-env occlusion levels."""
        self._occlusion_level[env_ids] = levels

    def update(self, step_counter: int) -> None:
        """Called every env step. Projects current frame and periodically aggregates."""
        # Project current LiDAR frame to spherical grid
        scan = self._project_to_spherical_grid()  # (N, V, H) in metres
        scan = self._process_scan(scan)  # (N, V, H) normalized [0, 1]

        # Push into history ring buffer
        self.hist_scans = torch.cat([self.hist_scans[:, 1:], scan.unsqueeze(1)], dim=1)

        # Aggregate on the update tick
        if step_counter % self.update_freq == 0:
            self._aggregated = torch.min(self.hist_scans, dim=1).values

    def get_depth_image(self) -> torch.Tensor:
        """Return the latest aggregated depth image, shape (num_envs, V, H), values in [0, 1].

        Random shape occlusions and per-env Perlin/full occlusion are sampled fresh and
        overlaid here (after the temporal min-aggregation) so they always appear in the
        output regardless of the history.
        """
        self._build_cage_mask(torch.arange(self.num_envs, device=self.device))
        out = torch.where(self._cage_mask, torch.ones_like(self._aggregated), self._aggregated)

        # Per-env occlusion level: Perlin-noise-based information loss
        # 0 = no occlusion, 1 = fully occluded, intermediate = Perlin holes
        occ = self._occlusion_level  # (N,)
        if (occ > 0.0).any():
            V, H = out.shape[1], out.shape[2]
            fill_values = torch.rand_like(out) * 0.1 + 0.9

            # Fully randomize Perlin state each call -> no temporal correlation.
            self._occlusion_perlin_z = torch.rand(self.num_envs, device=self.device) * 1.0e4
            P = self._occlusion_perm.shape[-1]
            self._occlusion_perm = self._sample_batched_permutations(self.num_envs, P)
            noise = perlin_2d_batch(
                num_envs=self.num_envs,
                height=V,
                width=H,
                scale=self._occlusion_perlin_scale,
                z_offsets=self._occlusion_perlin_z,
                perm=self._occlusion_perm,
                device=self.device,
            )
            noise_01 = (noise + 1.0) * 0.5  # map to [0, 1]

            full_mask = (occ == 1.0).unsqueeze(-1).unsqueeze(-1)
            partial = ((occ > 0.0) & (occ < 1.0)).unsqueeze(-1).unsqueeze(-1)
            threshold = (1.0 - occ).unsqueeze(-1).unsqueeze(-1)
            hole_mask = (noise_01 > threshold) & partial
            out = torch.where(full_mask | hole_mask, fill_values, out)

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_batched_permutations(self, batch_size: int, permutation_size: int) -> torch.Tensor:
        """Sample independent random permutations without a Python batch loop."""
        return torch.rand(batch_size, permutation_size, device=self.device).argsort(dim=-1)

    def _get_sensor_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Mid360 world-frame position and quaternion."""
        offset_pos = self._offset_pos.unsqueeze(0).expand(self.num_envs, -1)
        offset_rot = self._offset_rot.unsqueeze(0).expand(self.num_envs, -1)
        pos_w = self.robot.data.root_pos_w + math_utils.quat_apply(self.robot.data.root_quat_w, offset_pos)
        quat_w = math_utils.quat_mul(self.robot.data.root_quat_w, offset_rot)
        return pos_w, quat_w

    def _project_to_spherical_grid(self) -> torch.Tensor:
        """Project LiDAR hits into the spherical depth grid.

        Returns:
            grid: (num_envs, V, H) with distances in metres. Unfilled bins = max_distance.
        """
        pos_w, quat_w = self._get_sensor_pose_w()
        points_w = self.scanner.data.ray_hits_w  # (N, num_rays, 3)

        # Transform to sensor frame
        points_b = math_utils.quat_apply_inverse(
            quat_w.unsqueeze(1).expand(-1, points_w.shape[1], -1),
            points_w - pos_w.unsqueeze(1),
        )

        # Compute spherical coordinates
        r = torch.norm(points_b, dim=-1) + 1e-6
        valid_mask = torch.isfinite(r) & (r > 0.1)
        theta = torch.asin((points_b[:, :, 2] / r).clamp(-1.0, 1.0))
        phi = torch.atan2(points_b[:, :, 1], points_b[:, :, 0])

        # Bin assignment
        theta_bins = torch.round((theta - self._theta_low) / self._theta_res).long()
        phi_bins = torch.round((phi - self._phi_low) / self._phi_res).long()

        valid_mask &= (theta_bins >= 0) & (theta_bins < self.vertical_num_rays)
        valid_mask &= (phi_bins >= 0) & (phi_bins < self.horizontal_num_rays)

        theta_bins.clamp_(0, self.vertical_num_rays - 1)
        phi_bins.clamp_(0, self.horizontal_num_rays - 1)

        # Scatter-reduce (amin) into flat grid
        V, H = self.vertical_num_rays, self.horizontal_num_rays
        batch_idx = torch.arange(self.num_envs, device=self.device).unsqueeze(-1).expand(-1, points_w.shape[1])
        flat_idx = batch_idx * (V * H) + theta_bins * H + phi_bins

        grid = torch.full((self.num_envs * V * H,), self.max_distance, device=self.device)
        grid.scatter_reduce_(0, flat_idx[valid_mask], r[valid_mask], reduce="amin", include_self=True)
        return grid.view(self.num_envs, V, H)

    def _build_cage_mask(self, env_ids: torch.Tensor) -> None:
        """Build a random-shape occlusion mask for given environments.

        Each environment receives 0, 1, or 2 occluding shapes. Each shape is one of
        {sphere, rectangle, square, trapezoid} with a random center, random rotation
        in [0, 2*pi), and shape-specific random size:

          * sphere: radius in [3, 5] px
          * rectangle: length in [10, 30] px, width in [5, 10] px
          * square: side in [5, 10] px
          * trapezoid: top in [5, 10] px, bottom in [8, 14] px, height in [5, 10] px
        """
        V, H = self.vertical_num_rays, self.horizontal_num_rays
        n = env_ids.numel()
        max_shapes = 2
        param_shape = (n, max_shapes, 1, 1)

        num_shapes = torch.randint(0, max_shapes + 1, (n, 1), device=self.device)
        active_shapes = torch.arange(max_shapes, device=self.device).unsqueeze(0) < num_shapes
        shape_type = torch.randint(0, 4, (n, max_shapes), device=self.device)

        cr = torch.rand(param_shape, device=self.device) * V
        cc = torch.rand(param_shape, device=self.device) * H
        theta = torch.rand(param_shape, device=self.device) * (2.0 * math.pi)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        dr = self._grid_r[None, None] - cr
        dc = self._grid_c[None, None] - cc
        u = cos_t * dc + sin_t * dr
        v = -sin_t * dc + cos_t * dr

        radius = torch.rand(param_shape, device=self.device) * 2.0 + 3.0
        sphere_mask = (u * u + v * v) <= radius * radius

        length = torch.rand(param_shape, device=self.device) * 20.0 + 10.0
        width = torch.rand(param_shape, device=self.device) * 5.0 + 5.0
        rectangle_mask = (u.abs() <= length * 0.5) & (v.abs() <= width * 0.5)

        size = torch.rand(param_shape, device=self.device) * 5.0 + 5.0
        square_mask = (u.abs() <= size * 0.5) & (v.abs() <= size * 0.5)

        top_w = torch.rand(param_shape, device=self.device) * 5.0 + 5.0
        bot_w = torch.rand(param_shape, device=self.device) * 6.0 + 8.0
        height = torch.rand(param_shape, device=self.device) * 5.0 + 5.0
        t = (v + height * 0.5) / height
        half_w = (top_w * 0.5) * (1.0 - t) + (bot_w * 0.5) * t
        trapezoid_mask = (v.abs() <= height * 0.5) & (u.abs() <= half_w)

        type_masks = shape_type[..., None, None]
        shape_mask = torch.where(
            type_masks == 0,
            sphere_mask,
            torch.where(type_masks == 1, rectangle_mask, torch.where(type_masks == 2, square_mask, trapezoid_mask)),
        )
        self._cage_mask[env_ids] = (shape_mask & active_shapes[..., None, None]).any(dim=1)

    def _process_scan(self, scan: torch.Tensor) -> torch.Tensor:
        """Add noise, clip, normalize, and apply bottom-row dropout.

        Args:
            scan: (N, V, H) distances in metres.

        Returns:
            Processed scan, normalized to [0, 1].
        """
        # Gaussian noise
        scan = scan + torch.randn_like(scan) * self.noise_std
        scan = scan.clamp(0.0, self.max_distance) / self.max_distance

        # Bottom-row exponential dropout (emulates real Mid360 sparsity near the ground)
        V = scan.shape[1]
        dropout_start_row = int(V * (1.0 - self.bottom_dropout_fraction))
        num_dropout_rows = V - dropout_start_row
        if num_dropout_rows > 0:
            t = torch.arange(num_dropout_rows, device=self.device, dtype=torch.float32) / max(num_dropout_rows - 1, 1)
            dropout_prob = 0.5 * torch.exp(math.log(2.0) * t)
            full_prob = torch.zeros(V, device=self.device)
            full_prob[dropout_start_row:] = dropout_prob
            dropout_mask = torch.rand(scan.shape, device=self.device) < full_prob[None, :, None]
            scan = torch.where(dropout_mask, torch.ones_like(scan), scan)

        return scan
