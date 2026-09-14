"""Transformer-GRU Estimator for proprioceptive and depth-based state estimation.

Architecture (from diagram):
    Temporal Proprioceptive Observations → MLP → ┐
                                                  ├→ Transformer Encoder (+ CLS token) → GRU → Latent Heads
    Temporal Depth Images → CNN (per-channel) →   ┘

Latent heads:
    v_hat  (3)  — base linear velocity estimate
    hf_hat (4)  — foot clearance estimate
    z      (64) — pure latent
    zm     (64) — map latent

Decoders:
    next_prop_decoder:   [v_hat, hf_hat, z, zm] → next proprioception
    height_map_decoder:  zm → height map
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningNormalizer(nn.Module):
    """Tracks running mean/variance with exponential moving average for target normalization.

    Normalizes inputs to zero-mean, unit-variance for loss computation.
    Does NOT affect forward pass outputs — only used inside train_step.
    """

    def __init__(self, dim: int, momentum: float = 0.01, eps: float = 1e-6, var_min: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.var_min = var_min
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running stats from a batch. x: (*, dim)."""
        flat = x.reshape(-1, x.shape[-1])
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, correction=0)
        if not self.initialized:
            self.running_mean.copy_(batch_mean)
            self.running_var.copy_(batch_var.clamp(min=self.var_min))
            self.initialized.fill_(True)
        else:
            m = self.momentum
            self.running_mean.mul_(1 - m).add_(batch_mean, alpha=m)
            self.running_var.mul_(1 - m).add_(batch_var, alpha=m).clamp_(min=self.var_min)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x using running stats. x: (*, dim) → (*, dim)."""
        return (x - self.running_mean) / (self.running_var.sqrt() + self.eps)


class Estimator(nn.Module):
    """Transformer-GRU based estimator replacing the DreamerV3 world model."""

    def __init__(
        self,
        prop_dim: int = 45,
        num_prop_steps: int = 10,
        sensor_size: tuple = (64, 64, 2),
        base_lin_vel_dim: int = 3,
        foot_clearance_dim: int = 4,
        z_dim: int = 64,
        zm_dim: int = 64,
        next_prop_dim: int = 45,
        height_map_dim: int = 300,
        d_model: int = 512,
        nhead: int = 8,
        num_transformer_layers: int = 4,
        dim_feedforward: int = 1024,
        gru_hidden_dim: int = 512,
        lr: float = 2e-4,
        device: str = "cuda:0",
    ):
        super().__init__()

        self.prop_dim = prop_dim
        self.num_prop_steps = num_prop_steps
        self.sensor_size = sensor_size  # (H, W, C)
        self.num_depth_channels = sensor_size[2]
        self.base_lin_vel_dim = base_lin_vel_dim
        self.foot_clearance_dim = foot_clearance_dim
        self.z_dim = z_dim
        self.zm_dim = zm_dim
        self.next_prop_dim = next_prop_dim
        self.height_map_dim = height_map_dim
        self.d_model = d_model
        self.gru_hidden_dim = gru_hidden_dim
        self.device_str = device

        # Feature dim: [v_hat, hf_hat, z, zm]
        self.feature_dim = base_lin_vel_dim + foot_clearance_dim + z_dim + zm_dim

        # ---- Prop MLP: shared across timesteps ----
        self.prop_mlp = nn.Sequential(
            nn.Linear(prop_dim, 256),
            nn.ELU(),
            nn.Linear(256, d_model),
        )

        # ---- Depth CNN: per-channel, reduces 64x64 → 4x4 ----
        # 4 stride-2 conv layers: 64→32→16→8→4
        self.depth_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 64→32
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32→16
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16→8
            nn.ELU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),  # 8→4
            nn.ELU(),
        )
        self.depth_token_proj = nn.Linear(128, d_model)

        # ---- CLS token ----
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ---- Transformer Encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # ---- GRU ----
        self.gru = nn.GRUCell(input_size=d_model, hidden_size=gru_hidden_dim)
        # Learnable initial hidden state
        self.W_init = nn.Parameter(torch.randn(1, gru_hidden_dim) * 0.02)

        # ---- Latent heads (from GRU hidden) ----
        # v_hat: velocity
        self.v_head = nn.Linear(gru_hidden_dim, base_lin_vel_dim)

        # hf_hat: foot clearance
        self.hf_head = nn.Linear(gru_hidden_dim, foot_clearance_dim)

        # z: pure latent
        self.z_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, 256),
            nn.ELU(),
            nn.Linear(256, z_dim),
        )

        # zm: map latent
        self.zm_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, 256),
            nn.ELU(),
            nn.Linear(256, zm_dim),
        )

        # ---- Decoders ----
        # Next prop decoder: input = all 4 latent heads concatenated
        self.next_prop_decoder = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, next_prop_dim),
        )

        # Height map decoder: input = zm only
        self.height_map_decoder = nn.Sequential(
            nn.Linear(zm_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, height_map_dim),
        )

        # ---- Optimizer (separate from PPO) ----
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # ---- Running normalizers for prediction targets (used in train_step only) ----
        self.target_normalizers = nn.ModuleDict(
            {
                "next_prop": RunningNormalizer(next_prop_dim),
                "height_map": RunningNormalizer(height_map_dim),
                "velocity": RunningNormalizer(base_lin_vel_dim),
                "foot_clearance": RunningNormalizer(foot_clearance_dim),
            }
        )

        # Training step counter for logging
        self._train_step = 0

    def initial_hidden(self, batch_size: int) -> torch.Tensor:
        """Return learnable initial hidden state for GRU."""
        return torch.tanh(self.W_init).expand(batch_size, -1)

    def _encode_prop(self, prop_hist: torch.Tensor) -> torch.Tensor:
        """Encode proprioception history into tokens.

        Args:
            prop_hist: (B, num_prop_steps * prop_dim) flattened prop history

        Returns:
            tokens: (B, num_prop_steps, d_model)
        """
        B = prop_hist.shape[0]
        # Reshape to (B, T, prop_dim)
        prop = prop_hist.reshape(B, self.num_prop_steps, self.prop_dim)
        # Shared MLP across timesteps: (B, T, prop_dim) → (B, T, d_model)
        return self.prop_mlp(prop)

    def _encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Encode depth image(s) into spatial tokens, per-channel.

        Args:
            depth: (B, H, W, C) depth images

        Returns:
            tokens: (B, num_channels * 16, d_model)
        """
        B, H, W, C = depth.shape
        all_tokens = []
        for c in range(C):
            # Extract single channel: (B, 1, H, W)
            single = depth[..., c].unsqueeze(1)
            # CNN: (B, 1, 64, 64) → (B, 128, 4, 4)
            feat = self.depth_cnn(single)
            # Reshape to spatial tokens: (B, 128, 4, 4) → (B, 16, 128)
            feat = feat.flatten(2).transpose(1, 2)
            # Project to d_model: (B, 16, 128) → (B, 16, d_model)
            tokens = self.depth_token_proj(feat)
            all_tokens.append(tokens)
        # Concatenate all channels: (B, C*16, d_model)
        return torch.cat(all_tokens, dim=1)

    def forward(
        self,
        prop_hist: torch.Tensor,
        depth: torch.Tensor,
        hidden: torch.Tensor | None,
        is_first: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass of the estimator.

        Args:
            prop_hist: (B, num_prop_steps * prop_dim) proprioception history
            depth: (B, H, W, C) depth images
            hidden: (B, gru_hidden_dim) or None — GRU hidden state
            is_first: (B,) — 1.0 on episode reset, 0.0 otherwise

        Returns:
            feature: (B, feature_dim) concatenated latent heads [v, hf, z, zm]
            new_hidden: (B, gru_hidden_dim) updated GRU hidden
            preds: dict with v_hat, hf_hat, z, zm,
                   pred_next_prop, pred_height_map
        """
        B = prop_hist.shape[0]

        # Handle reset: blend hidden with initial state using is_first mask
        init_h = self.initial_hidden(B)
        if hidden is None:
            hidden = init_h
        else:
            mask = is_first.unsqueeze(-1)  # (B, 1)
            hidden = hidden * (1.0 - mask) + init_h * mask

        # Encode inputs
        prop_tokens = self._encode_prop(prop_hist)  # (B, 10, d_model)
        depth_tokens = self._encode_depth(depth)  # (B, 32, d_model)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)

        # Concatenate: [prop_tokens, depth_tokens, cls_token]
        tokens = torch.cat([prop_tokens, depth_tokens, cls], dim=1)  # (B, 43, d_model)

        # Transformer Encoder
        tokens = self.transformer_encoder(tokens)  # (B, 43, d_model)

        # Extract CLS token output (last position)
        cls_out = tokens[:, -1, :]  # (B, d_model)

        # GRU step
        new_hidden = self.gru(cls_out, hidden)  # (B, gru_hidden_dim)

        # Latent heads
        v_hat = self.v_head(new_hidden)
        hf_hat = self.hf_head(new_hidden)
        z = self.z_head(new_hidden)
        zm = self.zm_head(new_hidden)

        # Concatenate feature: [v_hat, hf_hat, z, zm]
        feature = torch.cat([v_hat, hf_hat, z, zm], dim=-1)

        # Decoders
        pred_next_prop = self.next_prop_decoder(feature)
        pred_height_map = self.height_map_decoder(zm)

        preds = {
            "v_hat": v_hat,
            "hf_hat": hf_hat,
            "z": z,
            "zm": zm,
            "pred_next_prop": pred_next_prop,
            "pred_height_map": pred_height_map,
        }

        return feature, new_hidden, preds

    def train_step(self, batch_data: dict) -> tuple[dict, dict | None]:
        """Train the estimator on a batch of sequences.

        Args:
            batch_data: dict with keys:
                - prop_hist:       (B, T, prop_hist_dim)
                - image:           (B, T, H, W, C)
                - curr_prop:       (B, T, prop_dim) — already shifted to be NEXT step
                - height_map:      (B, T, height_map_dim)
                - base_lin_vel:    (B, T, base_lin_vel_dim)
                - foot_clearance:  (B, T, foot_clearance_dim)
                - is_first:        (B, T)

        Returns:
            metrics: dict of scalar metrics
            height_map_samples: dict with "gt" and "pred" for visualization, or None
        """
        device = next(self.parameters()).device
        batch_data = {k: torch.as_tensor(v, device=device, dtype=torch.float32) for k, v in batch_data.items()}

        B, T = batch_data["prop_hist"].shape[:2]

        # Update running normalizers with all targets in this batch
        self.target_normalizers["next_prop"].update(batch_data["curr_prop"])
        self.target_normalizers["height_map"].update(batch_data["height_map"])
        self.target_normalizers["velocity"].update(batch_data["base_lin_vel"])
        self.target_normalizers["foot_clearance"].update(batch_data["foot_clearance"])

        # Run forward over sequence
        hidden = None
        total_next_prop_loss = 0.0
        total_height_map_loss = 0.0
        total_vel_loss = 0.0
        total_fc_loss = 0.0

        all_pred_height_maps = []
        all_preds = []

        for t in range(T):
            is_first_t = batch_data["is_first"][:, t]
            prop_hist_t = batch_data["prop_hist"][:, t]
            image_t = batch_data["image"][:, t]

            feature, hidden, preds = self.forward(prop_hist_t, image_t, hidden, is_first_t)

            # Losses
            # MSE: next prop reconstruction (normalized)
            next_prop_target = batch_data["curr_prop"][:, t]
            total_next_prop_loss += F.mse_loss(
                self.target_normalizers["next_prop"].normalize(preds["pred_next_prop"]),
                self.target_normalizers["next_prop"].normalize(next_prop_target),
            )

            # MSE: height map reconstruction (normalized)
            hm_target = batch_data["height_map"][:, t]
            total_height_map_loss += F.mse_loss(
                self.target_normalizers["height_map"].normalize(preds["pred_height_map"]),
                self.target_normalizers["height_map"].normalize(hm_target),
            )

            # MSE: velocity supervision (normalized)
            vel_target = batch_data["base_lin_vel"][:, t]
            total_vel_loss += F.mse_loss(
                self.target_normalizers["velocity"].normalize(preds["v_hat"]),
                self.target_normalizers["velocity"].normalize(vel_target),
            )

            # MSE: foot clearance supervision (normalized)
            fc_target = batch_data["foot_clearance"][:, t]
            total_fc_loss += F.mse_loss(
                self.target_normalizers["foot_clearance"].normalize(preds["hf_hat"]),
                self.target_normalizers["foot_clearance"].normalize(fc_target),
            )

            all_pred_height_maps.append(preds["pred_height_map"].detach())
            all_preds.append({k: v.detach() for k, v in preds.items()})

        # Average over time
        total_next_prop_loss /= T
        total_height_map_loss /= T
        total_vel_loss /= T
        total_fc_loss /= T

        # Total loss
        loss = total_next_prop_loss + total_height_map_loss + total_vel_loss + total_fc_loss

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        # Metrics
        metrics = {
            "next_prop_loss": total_next_prop_loss.item(),
            "height_map_loss": total_height_map_loss.item(),
            "velocity_loss": total_vel_loss.item(),
            "foot_clearance_loss": total_fc_loss.item(),
            "total_loss": loss.item(),
        }

        # Height map samples for visualization
        height_map_samples = None
        num_samples = min(4, B)
        sample_idx = torch.randperm(B, device=device)[:num_samples]
        height_map_samples = {
            "gt": batch_data["height_map"][sample_idx, -1].detach(),
            "pred": all_pred_height_maps[-1][sample_idx].detach(),
        }

        # Console logging
        self._train_step += 1
        if self._train_step % 50 == 0:
            prop_gt = batch_data["curr_prop"][0, -1].detach().cpu().numpy()
            prop_pred = all_preds[-1]["pred_next_prop"][0].cpu().numpy()
            prop_err = prop_gt - prop_pred
            print(f"[Estimator next_prop] GT:   {' '.join(f'{x:+.3f}' for x in prop_gt)}")
            print(f"[Estimator next_prop] Pred: {' '.join(f'{x:+.3f}' for x in prop_pred)}")
            print(
                f"[Estimator next_prop] Err:  {' '.join(f'{x:+.3f}' for x in prop_err)}  "
                f"(MSE={float((prop_err**2).mean()):.5f})"
            )
            print(f"[Estimator] Losses: {', '.join(f'{k}={v:.5f}' for k, v in metrics.items())}")
            print()

        return metrics, height_map_samples
