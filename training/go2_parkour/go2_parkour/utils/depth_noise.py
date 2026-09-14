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

"""Depth image degradation for sim-to-real transfer.

Applies realistic noise to simulated depth images to mimic the imperfections of
real stereo-matching depth cameras (e.g. Intel RealSense D435i). The pipeline is:

    1. Clip  – enforce near/far range; pixels < near_clip are set to max_distance.
    2. Edge noise – pixels near depth discontinuities are randomly emptied or shuffled.
    3. Holes – slowly-evolving Perlin-noise-based patches are set to max_distance.
    4. Gaussian blur – smooth the entire image with a Gaussian kernel.

Reference: "Extreme Parkour with Legged Robots" (Cheng et al.), Fig. 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class DepthNoiseCfg:
    """Configuration for simulated depth-image degradation."""

    enabled: bool = True
    """Master toggle for all depth noise."""

    # -- 1. clip ---------------------------------------------------------------
    near_clip: float = 0.15
    """Minimum valid depth (m). Pixels below this are set to max_distance."""

    # -- 2. edge noise ---------------------------------------------------------
    edge_threshold: float = 0.025
    """Laplacian-magnitude threshold (m) to detect depth discontinuities.
    Uses a second-order derivative so smooth angled surfaces are ignored."""

    edge_dilate_kernel: int = 3
    """Square kernel size for dilating the edge mask (pixels)."""

    edge_erase_prob: float = 0.3
    """Probability that an identified edge pixel is set to empty (max_distance).
    The remaining edge pixels are shuffled with a random neighbor."""

    # -- 3. holes (Perlin noise) -----------------------------------------------
    perlin_scale: float = 8.0
    """Spatial frequency of the Perlin noise field (number of octaves spanning
    the image). Larger → smaller patches."""

    perlin_threshold: float = 0.72
    """Perlin noise values above this become holes.  Larger → fewer holes."""

    perlin_time_scale: float = 0.01
    """Per-step increment of the Perlin noise time/z-offset, controlling how
    fast the hole pattern evolves."""

    # -- 4. gaussian blur ------------------------------------------------------
    blur_kernel_size: int = 5
    """Size of the Gaussian blur kernel (must be odd)."""

    blur_sigma: float = 1.0
    """Standard deviation of the Gaussian blur."""


# ── Pure-PyTorch 2-D Perlin Noise ───────────────────────────────────────────


def _fade(t: torch.Tensor) -> torch.Tensor:
    """Perlin fade / smootherstep: 6t^5 − 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return a + t * (b - a)


def _hash_2d(ix: torch.Tensor, iy: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Double-hash integer coordinates through a random permutation table.

    Args:
        ix, iy: integer grid coordinates – any shape (broadcastable).
        perm: (num_envs, P) random permutation LUT.

    Returns:
        Index into perm, same shape as ix/iy, values in [0, P).
    """
    P = perm.shape[-1]
    # perm is (num_envs, P); ix/iy are (num_envs, H, W) or broadcastable
    idx = perm.gather(1, (ix % P).reshape(ix.shape[0], -1)).reshape(ix.shape)
    idx = perm.gather(1, ((idx + iy) % P).reshape(ix.shape[0], -1)).reshape(ix.shape)
    return idx


def _grad_2d(hash_val: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Map a hash value to one of 4 gradient directions and dot with (dx, dy).

    Gradients: (1,1), (-1,1), (1,-1), (-1,-1).
    """
    h = hash_val & 3  # 0..3
    u = torch.where(h < 2, dx, -dx)
    v = torch.where((h == 0) | (h == 3), dy, -dy)
    return u + v


def perlin_2d_batch(
    num_envs: int,
    height: int,
    width: int,
    scale: float,
    z_offsets: torch.Tensor,
    perm: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Generate a batch of 2-D Perlin noise fields on the GPU.

    The ``z_offsets`` parameter shifts the *y* coordinate continuously so the
    pattern evolves over time while remaining spatially coherent.

    Args:
        num_envs: batch size.
        height, width: spatial dimensions of the output.
        scale: how many noise cells span the image (higher → finer detail).
        z_offsets: (num_envs,) per-env temporal offset (float).
        perm: (num_envs, P) random permutation table.
        device: torch device.

    Returns:
        (num_envs, height, width) noise in approximately [-1, 1].
    """
    # Coordinate grids – shape (1, H, W), will broadcast with (N, 1, 1).
    ys = torch.linspace(0, scale, height, device=device).unsqueeze(1).expand(height, width).unsqueeze(0)
    xs = torch.linspace(0, scale, width, device=device).unsqueeze(0).expand(height, width).unsqueeze(0)

    # Add per-env temporal offset to y
    ys = ys + z_offsets[:, None, None]  # (N, H, W)
    xs = xs.expand(num_envs, -1, -1)  # (N, H, W)

    # Integer and fractional parts
    ix0 = xs.long()
    iy0 = ys.long()
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    fx = xs - ix0.float()
    fy = ys - iy0.float()

    u = _fade(fx)
    v = _fade(fy)

    # Hashed gradient indices at four corners
    n00 = _grad_2d(_hash_2d(ix0, iy0, perm), fx, fy)
    n10 = _grad_2d(_hash_2d(ix1, iy0, perm), fx - 1.0, fy)
    n01 = _grad_2d(_hash_2d(ix0, iy1, perm), fx, fy - 1.0)
    n11 = _grad_2d(_hash_2d(ix1, iy1, perm), fx - 1.0, fy - 1.0)

    # Bilinear interpolation
    nx0 = _lerp(n00, n10, u)
    nx1 = _lerp(n01, n11, u)
    return _lerp(nx0, nx1, v)


# ── Gaussian Kernel ─────────────────────────────────────────────────────────


def _gaussian_kernel_2d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2-D Gaussian kernel of shape (1, 1, K, K)."""
    ax = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)


# ── Laplacian Kernel ────────────────────────────────────────────────────────


def _laplacian_kernel(device: torch.device) -> torch.Tensor:
    """Return a 3×3 Laplacian kernel of shape (1, 1, 3, 3).

    The Laplacian is a second-order derivative that responds to depth
    *discontinuities* (step edges) but is approximately zero on smooth
    surfaces — even when those surfaces are angled relative to the camera.
    """
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device)
    return lap.unsqueeze(0).unsqueeze(0)


# ── Main Degrader Class ─────────────────────────────────────────────────────


class DepthImageDegrader:
    """Applies a sim-to-real degradation pipeline to batched depth images.

    All operations run on the GPU with no Python loops over environments.
    """

    def __init__(
        self,
        cfg: DepthNoiseCfg,
        num_envs: int,
        height: int,
        width: int,
        max_distance: float,
        device: torch.device | str,
    ):
        self.cfg = cfg
        self.num_envs = num_envs
        self.height = height
        self.width = width
        self.max_distance = max_distance
        self.device = torch.device(device)

        if not cfg.enabled:
            return

        # Pre-compute static kernels
        self._gauss_kernel = _gaussian_kernel_2d(cfg.blur_kernel_size, cfg.blur_sigma, self.device)
        self._laplacian = _laplacian_kernel(self.device)

        # Perlin noise state
        P = 256  # permutation table size
        self._perm = torch.stack([torch.randperm(P, device=self.device) for _ in range(num_envs)])  # (num_envs, P)
        self._perlin_z = torch.zeros(num_envs, device=self.device)

    # --------------------------------------------------------------------- #

    def degrade(self, depth_image: torch.Tensor) -> torch.Tensor:
        """Apply the full degradation pipeline.

        Args:
            depth_image: (num_envs, H, W) raw depth in *meters*.

        Returns:
            Degraded depth in meters, same shape.
        """
        if not self.cfg.enabled:
            return depth_image

        img = depth_image.clone()

        # 1. Clip ─────────────────────────────────────────────────────────────
        img = torch.clamp(img, max=self.max_distance)
        img[img < self.cfg.near_clip] = self.max_distance

        # 2. Edge noise ───────────────────────────────────────────────────────
        img = self._apply_edge_noise(img)

        # 3. Perlin holes ─────────────────────────────────────────────────────
        img = self._apply_perlin_holes(img)

        # 4. Gaussian blur ────────────────────────────────────────────────────
        img = self._apply_gaussian_blur(img)

        return img

    # --------------------------------------------------------------------- #

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset Perlin time offsets for the given environments."""
        if not self.cfg.enabled:
            return
        self._perlin_z[env_ids] = 0.0
        # Re-seed permutation tables so each episode gets a fresh pattern
        P = self._perm.shape[-1]
        for idx in env_ids:
            self._perm[idx] = torch.randperm(P, device=self.device)

    # ── Private helpers ─────────────────────────────────────────────────── #

    def _apply_edge_noise(self, img: torch.Tensor) -> torch.Tensor:
        """Detect depth discontinuities via Laplacian and corrupt nearby pixels.

        The Laplacian (second-order derivative) is ~0 on smooth angled surfaces
        where depth changes linearly, but large at actual depth step-edges.
        """
        N = img.shape[0]
        # (N, 1, H, W) for conv2d – use replicate padding so border pixels
        # don't see a fake depth-to-zero jump that would trigger false edges.
        x = img.unsqueeze(1)
        x_padded = F.pad(x, [1, 1, 1, 1], mode="replicate")

        lap = F.conv2d(x_padded, self._laplacian)
        lap_mag = lap.abs().squeeze(1)  # (N, H, W)

        edge_mask = lap_mag > self.cfg.edge_threshold

        # Dilate the edge mask
        k = self.cfg.edge_dilate_kernel
        if k > 1:
            edge_dilated = (
                F.max_pool2d(edge_mask.float().unsqueeze(1), kernel_size=k, stride=1, padding=k // 2).squeeze(1).bool()
            )
        else:
            edge_dilated = edge_mask

        # For each edge pixel, either erase or shuffle with a random neighbor
        erase = torch.rand(N, self.height, self.width, device=self.device) < self.cfg.edge_erase_prob
        erase_mask = edge_dilated & erase

        # Shuffle: shift image by a small random offset and use those values
        # We approximate "shuffle with random neighbor" by rolling in a random
        # direction for all non-erased edge pixels.
        shift_h = torch.randint(-1, 2, (1,), device=self.device).item()
        shift_w = torch.randint(-1, 2, (1,), device=self.device).item()
        shuffled = torch.roll(img, shifts=(shift_h, shift_w), dims=(1, 2))

        shuffle_mask = edge_dilated & ~erase

        img = torch.where(erase_mask, torch.full_like(img, self.max_distance), img)
        img = torch.where(shuffle_mask, shuffled, img)

        return img

    def _apply_perlin_holes(self, img: torch.Tensor) -> torch.Tensor:
        """Punch temporally-coherent holes using Perlin noise."""
        noise = perlin_2d_batch(
            num_envs=self.num_envs,
            height=self.height,
            width=self.width,
            scale=self.cfg.perlin_scale,
            z_offsets=self._perlin_z,
            perm=self._perm,
            device=self.device,
        )
        # noise is roughly in [-1, 1]; map to [0, 1] for thresholding
        noise_01 = (noise + 1.0) * 0.5
        hole_mask = noise_01 > self.cfg.perlin_threshold

        img = torch.where(hole_mask, torch.full_like(img, self.max_distance), img)

        # Advance the temporal offset
        self._perlin_z += self.cfg.perlin_time_scale

        return img

    def _apply_gaussian_blur(self, img: torch.Tensor) -> torch.Tensor:
        """Blur the depth image with a pre-computed Gaussian kernel."""
        k = self.cfg.blur_kernel_size
        p = k // 2
        # (N, 1, H, W) – replicate-pad to avoid darkening borders
        x = img.unsqueeze(1)
        x_padded = F.pad(x, [p, p, p, p], mode="replicate")
        blurred = F.conv2d(x_padded, self._gauss_kernel)
        return blurred.squeeze(1)
