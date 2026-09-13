from __future__ import annotations

import torch
from torch import nn

from rsl_rl.modules.actor_critic_teacher import ActorCriticTeacher


class StudentJepaEncoder(nn.Module):
    HIGH_FEATURE_HIDDEN_DIMS = (512, 256)
    HIGH_FEATURE_LATENT_DIM = 128
    PROPRIO_HIDDEN_DIMS = (512, 256)
    PROPRIO_LATENT_DIM = 128
    FUSION_HIDDEN_DIMS = (512, 256)

    def __init__(
        self,
        obs_proprio_dim,
        num_obs_hist,
        high_feature_dim,
        output_dims,
    ):
        super().__init__()

        self.obs_proprio_dim = obs_proprio_dim
        self.num_obs_hist = num_obs_hist
        self.proprio_hist_dim = obs_proprio_dim * num_obs_hist
        self.high_feature_dim = high_feature_dim

        self.high_feat_encoder = self._build_mlp(
            input_dim=high_feature_dim,
            hidden_dims=self.HIGH_FEATURE_HIDDEN_DIMS,
            output_dim=self.HIGH_FEATURE_LATENT_DIM,
        )
        self.proprio_encoder = self._build_mlp(
            input_dim=self.proprio_hist_dim,
            hidden_dims=self.PROPRIO_HIDDEN_DIMS,
            output_dim=self.PROPRIO_LATENT_DIM,
        )
        self.fusion = self._build_backbone(
            input_dim=self.PROPRIO_LATENT_DIM + self.HIGH_FEATURE_LATENT_DIM,
            hidden_dims=self.FUSION_HIDDEN_DIMS,
        )
        self.latent_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(self.FUSION_HIDDEN_DIMS[-1], output_dim), nn.Tanh())
            for output_dim in output_dims
        )

    @staticmethod
    def _build_mlp(input_dim, hidden_dims, output_dim):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SiLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_backbone(input_dim, hidden_dims):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SiLU())
            prev_dim = hidden_dim
        return nn.Sequential(*layers)

    def forward(self, proprio_hist, high_feat):
        if proprio_hist.shape[-1] != self.proprio_hist_dim:
            raise ValueError(f"proprio_hist last dim must be {self.proprio_hist_dim}, got {proprio_hist.shape[-1]}.")

        leading_shape = proprio_hist.shape[:-1]
        feature_shape = (*leading_shape, self.high_feature_dim)
        if tuple(high_feat.shape) != feature_shape:
            raise ValueError(f"high_feat shape must be {feature_shape}, got {tuple(high_feat.shape)}.")

        high_latent = self.high_feat_encoder(high_feat)
        proprio_latent = self.proprio_encoder(proprio_hist)
        fused = self.fusion(torch.cat([proprio_latent, high_latent], dim=-1))
        return torch.cat([head(fused) for head in self.latent_heads], dim=-1)


class ActorCriticTeacherCTS(ActorCriticTeacher):
    def __init__(
        self,
        num_actions,
        env_cfg,
        jepa_feature_dim,
        student_proprio_dim=45,
        **kwargs,
    ):
        super().__init__(num_actions=num_actions, env_cfg=env_cfg, **kwargs)

        self.obs_proprio_hist_range = env_cfg.obs_proprio_hist_range
        self.num_obs_hist = env_cfg.num_obs_hist
        self.student_proprio_dim = student_proprio_dim
        self.student_proprio_hist_dim = self.student_proprio_dim * self.num_obs_hist
        self.jepa_feature_dim = jepa_feature_dim
        self.teacher_latent_dims = (
            self.proprio_hist_encode_dim,
            self.height_map_encode_dim,
            self.priv_encode_dim,
        )
        self.student_encoder = StudentJepaEncoder(
            obs_proprio_dim=self.student_proprio_dim,
            num_obs_hist=self.num_obs_hist,
            high_feature_dim=self.jepa_feature_dim,
            output_dims=self.teacher_latent_dims,
        )

        print("=" * 30)
        print("ActorCriticTeacherCTS student encoder:")
        print(self.student_encoder)
        print("=" * 30)

    def extract_student_proprio_hist(self, observations):
        hist = observations[..., self.obs_proprio_hist_range[0] : self.obs_proprio_hist_range[1]]
        hist = hist.reshape(*hist.shape[:-1], self.num_obs_hist, self.obs_proprio_dim)
        return hist[..., : self.student_proprio_dim].reshape(*hist.shape[:-2], self.student_proprio_hist_dim)

    def encode_student_latent(self, observations, high_feat):
        proprio_hist = self.extract_student_proprio_hist(observations)
        return self.student_encoder(proprio_hist, high_feat)

    def encode_teacher_latent(self, observations):
        return torch.cat(
            [
                self.proprio_hist_encoder(
                    observations[..., self.obs_proprio_hist_range[0] : self.obs_proprio_hist_range[1]]
                ),
                self.height_map_encoder(
                    observations[..., self.obs_height_scan_range[0] : self.obs_height_scan_range[1]]
                ),
                self.priv_encoder(observations[..., self.obs_priv_range[0] : self.obs_priv_range[1]]),
            ],
            dim=-1,
        )

    def get_student_actor_obs(self, observations, high_feat, detach_latent=False):
        student_latent = self.encode_student_latent(observations, high_feat)
        if detach_latent:
            student_latent = student_latent.detach()
        return torch.cat(
            [
                observations[..., self.obs_proprio_range[0] : self.obs_proprio_range[1]],
                student_latent,
            ],
            dim=-1,
        )

    def act_teacher(self, observations):
        return super().act(observations)

    def act_student(self, observations, high_feat):
        self.update_distribution(self.get_student_actor_obs(observations, high_feat, detach_latent=True))
        return self.distribution.sample()

    def act_inference_student(self, observations, high_feat):
        return self.actor(self.get_student_actor_obs(observations, high_feat))

    def evaluate_teacher(self, observations):
        return super().evaluate(observations)

    def evaluate_student(self, observations, high_feat):
        return super().evaluate(observations)

    def ppo_parameters(self):
        params = []
        params.extend(self.actor.parameters())
        params.extend(self.critic.parameters())
        params.extend(self.proprio_hist_encoder.parameters())
        params.extend(self.height_map_encoder.parameters())
        params.extend(self.priv_encoder.parameters())
        if hasattr(self, "std"):
            params.append(self.std)
        if hasattr(self, "log_std"):
            params.append(self.log_std)
        return params

    def student_encoder_parameters(self):
        return self.student_encoder.parameters()
