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

import copy
import math

import torch
import torch.distributions as torchd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

import lejepa


def build_2d_sincos_position_embedding(grid_h: int, grid_w: int, embed_dim: int) -> torch.Tensor:
    """Standard 2D sin-cos positional embedding (as in MoCo-v3 / MAE).

    Returns a tensor of shape (grid_h * grid_w, embed_dim).
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos pos embed"
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32),
        indexing="ij",
    )
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (10000**omega)  # (pos_dim,)

    out_x = torch.einsum("m,d->md", grid_x.flatten(), omega)  # (N, pos_dim)
    out_y = torch.einsum("m,d->md", grid_y.flatten(), omega)  # (N, pos_dim)
    pos_embed = torch.cat(
        [torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=1
    )  # (N, embed_dim)
    return pos_embed


class _ViTBlock(nn.Module):
    """Pre-norm transformer block with QK-norm and GELU MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        # QK-norm: per-head LayerNorm on Q and K (improves from-scratch stability of small ViTs).
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, N, H, Dh)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # (B, H, N, Dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, H, N, Dh)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class SmallViT(nn.Module):
    """Small Vision Transformer (~1M params) for depth-image encoding.

    Best-practice choices:
      - Conv-based patch embedding.
      - Fixed 2D sin-cos positional embedding (no learned pos params).
      - Pre-norm transformer blocks with GELU MLP.
      - QK-norm for from-scratch stability.
      - Learned [CLS] token for pooling (LeWM-style).
    """

    def __init__(
        self,
        in_channels: int,
        img_size: tuple,
        patch_size: tuple,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
        out_dim: int = 128,
    ):
        super().__init__()
        H, W = img_size
        pH, pW = patch_size
        assert H % pH == 0 and W % pW == 0, "img_size must be divisible by patch_size"
        self.grid_h = H // pH
        self.grid_w = W // pW
        self.num_tokens = self.grid_h * self.grid_w

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learned [CLS] token (LeWM/ViT-style); patch tokens use fixed 2D sin-cos pos embed,
        # the CLS token gets its own learned positional embedding.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

        pos_embed = build_2d_sincos_position_embedding(self.grid_h, self.grid_w, embed_dim)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)  # (1, N, D)

        self.blocks = nn.ModuleList([_ViTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # Projection head: 1-layer MLP with BatchNorm (LeWM-style).
        # The final LayerNorm in the ViT prevents anti-collapse objectives from being
        # optimized effectively, so we project into a new representation space using
        # a Linear layer followed by BatchNorm1d before producing the latent.
        self.head = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, D, gH, gW)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = x + self.pos_embed
        cls_tokens = (self.cls_token + self.cls_pos_embed).expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1 + N, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 0]  # CLS token -> (B, D)
        return self.head(x)


class JepaEstimatorSensorSS(nn.Module):
    """SS as in Self-Supervised."""

    depth_latent_dim = 64
    prop_hist_latent_dim = 32
    action_hist_latent_dim = 32
    latent_pred_hidden_dim = 512

    raw_depth_shape = (25, 60)
    depth_shape = (32, 64)

    def __init__(
        self,
        temporal_steps,
        num_one_step_obs,
        prop_hist_dim,
        num_depth_channels,
        action_hist_dim,
        activation="elu",
        max_grad_norm=10.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "JepaEstimatorSensor.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(JepaEstimatorSensorSS, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.prop_hist_dim = prop_hist_dim
        self.num_depth_channels = num_depth_channels
        self.action_hist_dim = action_hist_dim
        self.max_grad_norm = max_grad_norm

        self.depth_encoder = SmallViT(
            in_channels=self.num_depth_channels,
            img_size=self.depth_shape,
            patch_size=(4, 8),
            embed_dim=128,
            depth=6,
            num_heads=4,
            mlp_ratio=2.0,
            out_dim=self.depth_latent_dim,
        )

        self.prop_hist_encoder = nn.Sequential(
            nn.Linear(self.prop_hist_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, self.prop_hist_latent_dim),
        )

        self.action_hist_encoder = nn.Sequential(
            nn.Linear(self.action_hist_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.action_hist_latent_dim),
        )

        self.pred_gru = nn.GRUCell(
            input_size=self.depth_latent_dim + self.prop_hist_latent_dim,
            hidden_size=self.latent_pred_hidden_dim,
        )

        predictor_input_dim = self.latent_pred_hidden_dim + self.action_hist_latent_dim
        self.latent_predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, self.depth_latent_dim),
        )

        self.prop_latent_predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, self.prop_hist_latent_dim),
        )

        # SIGReg loss
        univariate_test = lejepa.univariate.EppsPulley()
        self.sigreg_loss = lejepa.multivariate.SlicingUnivariateTest(univariate_test=univariate_test, num_slices=512)

        # Optimizer
        self.learning_rate = 5e-4
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        print("__init__ of JepaEstimatorSensorSS:")
        print(self)

    def get_feature_dim(self):
        return self.latent_pred_hidden_dim

    def get_hidden_dim(self):
        return self.latent_pred_hidden_dim

    def pad_depth(self, depth):
        """
        depth: (B, num_depth_channels, raw_depth_shape[0], raw_depth_shape[1])
        -> padded_depth: (B, num_depth_channels, depth_shape[0], depth_shape[1])
        """
        pad_h = self.depth_shape[0] - self.raw_depth_shape[0]  # 32 - 25 = 7
        pad_w = self.depth_shape[1] - self.raw_depth_shape[1]  # 64 - 60 = 4
        # F.pad order: (left, right, top, bottom)
        return F.pad(depth, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

    def get_depth_shape(self):
        return (self.num_depth_channels,) + self.raw_depth_shape

    def forward(self, prop_hist: torch.Tensor, depth_stack: torch.Tensor, hidden: torch.Tensor):
        """
        Inputs:
            prop_hist: (B, prop_hist_dim)
            depth_stack: (B, num_depth_channels, raw_depth_shape[0], raw_depth_shape[1])
            hidden: (B, latent_pred_hidden_dim)

        Returns:
            feature: (B, latent_pred_hidden_dim)
            hidden: (B, latent_pred_hidden_dim)
        """

        prop_hist_latent = self.prop_hist_encoder(prop_hist)  # (B, prop_hist_latent_dim)

        depth_stack = self.pad_depth(depth_stack)  # (B, num_depth_channels, depth_shape[0], depth_shape[1])
        depth_latent = self.depth_encoder(depth_stack)  # (B, depth_latent_dim)

        latent = torch.cat([prop_hist_latent, depth_latent], dim=-1)  # (B, prop_hist_latent_dim + depth_latent_dim)

        pred_hidden = self.pred_gru(latent, hidden)  # (B, latent_pred_hidden_dim)

        return pred_hidden.detach(), pred_hidden.detach()

    def update(self, batch_data, lr=None):
        """
        Inputs:
            prop_hist: (T, B, prop_hist_dim)
            depth_stack: (T, B, num_depth_channels, raw_depth_shape[0], raw_depth_shape[1])
            action_hist: (T, B, action_dim)
            hidden_states: (T, B, latent_pred_hidden_dim)
            dones: (T, B, 1)

        Returns:
            loss_dict: dict of losses
        """
        # if lr is not None:
        #     self.learning_rate = lr
        #     for param_group in self.optimizer.param_groups:
        #         param_group["lr"] = self.learning_rate

        prop_hists: torch.Tensor = batch_data["prop_hist"]
        depth_stacks: torch.Tensor = batch_data["depth_stack"]
        action_hists: torch.Tensor = batch_data["action_hist"]
        hidden_states: torch.Tensor = batch_data["hidden_states"]
        dones: torch.Tensor = batch_data["dones"]

        prop_hists = prop_hists.detach()
        depth_stacks = depth_stacks.detach()
        action_hists = action_hists.detach()
        hidden_states = hidden_states.detach()
        dones = dones.detach()

        pred_hiddens = hidden_states  # (T, B, latent_pred_hidden_dim)

        # Encode proprioception history
        prop_hist_latents = self.prop_hist_encoder(prop_hists)  # (T, B, prop_hist_latent_dim)

        # Encode depth stacks
        T, B, _, _, _ = depth_stacks.shape
        depth_stacks = depth_stacks.flatten(0, 1)  # (T * B, num_depth_channels, raw_depth_shape[0], raw_depth_shape[1])
        depth_stacks = self.pad_depth(depth_stacks)  # (T * B, num_depth_channels, depth_shape[0], depth_shape[1])
        depth_latents = self.depth_encoder(depth_stacks)  # (T * B, depth_latent_dim)
        depth_latents = torch.unflatten(depth_latents, 0, (T, B))  # (T, B, depth_latent_dim)

        # Encode action history
        action_hist_latents = self.action_hist_encoder(action_hists)  # (T, B, action_hist_latent_dim)

        # Aggregate sensor latents and predict the next depth/proprio latents.
        # action_hists[t] stores the action block after estimator state t,
        # so predicting t -> t + 1 uses action_hists[t].
        total_latent_pred_loss = torch.tensor(0.0, device=depth_latents.device)
        total_prop_latent_pred_loss = torch.tensor(0.0, device=depth_latents.device)
        pred_hidden = pred_hiddens[0]  # (B, latent_pred_hidden_dim)
        num_prediction_terms = 0

        for t in range(T - 1):
            if t > 0:
                reset_mask = dones[t - 1].squeeze(-1).bool()  # (B,)
                if reset_mask.any():
                    pred_hidden = torch.where(reset_mask.unsqueeze(-1), pred_hiddens[t], pred_hidden)

            latent = torch.cat(
                [prop_hist_latents[t], depth_latents[t]], dim=-1
            )  # (B, prop_hist_latent_dim + depth_latent_dim)

            # Update latent predictor GRU and predict next step's latent
            pred_hidden = self.pred_gru(latent, pred_hidden)  # (B, latent_pred_hidden_dim)
            predictor_input = torch.cat([pred_hidden, action_hist_latents[t]], dim=-1)
            predicted_next_latent = self.latent_predictor(predictor_input)  # (B, depth_latent_dim)
            predicted_next_prop_latent = self.prop_latent_predictor(predictor_input)  # (B, prop_hist_latent_dim)

            valid_mask = ~dones[t].squeeze(-1).bool()  # (B,)
            if valid_mask.any():
                total_latent_pred_loss += F.mse_loss(
                    predicted_next_latent[valid_mask], depth_latents[t + 1][valid_mask]
                )
                total_prop_latent_pred_loss += F.mse_loss(
                    predicted_next_prop_latent[valid_mask], prop_hist_latents[t + 1][valid_mask]
                )
                num_prediction_terms += 1

        denom = max(num_prediction_terms, 1)
        total_latent_pred_loss /= denom
        total_prop_latent_pred_loss /= denom
        sigreg_loss = self.sigreg_loss(depth_latents)  # (T, B, depth_latent_dim)
        prop_sigreg_loss = self.sigreg_loss(prop_hist_latents)  # (T, B, prop_hist_latent_dim)

        total_loss = total_latent_pred_loss + total_prop_latent_pred_loss + 0.1 * sigreg_loss + 0.1 * prop_sigreg_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "latent_pred_loss": total_latent_pred_loss.item(),
            "prop_latent_pred_loss": total_prop_latent_pred_loss.item(),
            "sigreg_loss": sigreg_loss.item(),
            "prop_sigreg_loss": prop_sigreg_loss.item(),
            "total_loss": total_loss.item(),
        }

    @staticmethod
    def vicreg_loss(z):
        # z: (T, B, D) → flatten to (T*B, D)
        z = z.reshape(-1, z.shape[-1])

        # Variance: encourage std of each dimension ≥ 1
        std = z.std(dim=0)
        var_loss = torch.relu(1.0 - std).mean()

        # Covariance: decorrelate dimensions
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (z.shape[0] - 1)
        off_diag = cov - torch.diag(cov.diag())
        cov_loss = off_diag.pow(2).mean()

        return var_loss + cov_loss


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
