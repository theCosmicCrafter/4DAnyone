"""Canonical Deformable Gaussian Splatting (Deformable-GS) model implementation.

Features an 8-layer sinusoidal deformation MLP with skip connections, zero-initialized delta heads,
and canonical 3D Gaussians initialized from visual-hull sparse point clouds.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger("fdanyone.deformable_gs")


class SinusoidalEncoder(nn.Module):
    """Positional encoding with power-of-two frequency bands."""

    def __init__(self, in_channels: int, num_frequencies: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_frequencies = num_frequencies
        freq_bands = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    @property
    def output_dim(self) -> int:
        return self.in_channels * self.num_frequencies * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, C] -> [N, C, F] -> [N, C * F * 2]
        scaled = x.unsqueeze(-1) * self.freq_bands * math.pi
        sin_features = torch.sin(scaled)
        cos_features = torch.cos(scaled)
        features = torch.cat([sin_features, cos_features], dim=-1)
        return features.view(x.shape[0], -1)


class DeformationMLP(nn.Module):
    """8-layer deformation network with skip connection predicting position, rotation, and scale offsets."""

    def __init__(
        self,
        spatial_frequencies: int = 10,
        temporal_frequencies: int = 6,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.pos_encoder = SinusoidalEncoder(3, spatial_frequencies)
        self.time_encoder = SinusoidalEncoder(1, temporal_frequencies)
        in_dim = self.pos_encoder.output_dim + self.time_encoder.output_dim

        self.block1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Output heads with zero initialization for exact identity initial state
        self.delta_pos = nn.Linear(hidden_dim, 3)
        self.delta_rot = nn.Linear(hidden_dim, 4)
        self.delta_scale = nn.Linear(hidden_dim, 3)

        nn.init.zeros_(self.delta_pos.weight)
        nn.init.zeros_(self.delta_pos.bias)
        nn.init.zeros_(self.delta_rot.weight)
        nn.init.zeros_(self.delta_rot.bias)
        nn.init.zeros_(self.delta_scale.weight)
        nn.init.zeros_(self.delta_scale.bias)

    def forward(self, means: torch.Tensor, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute delta position, quaternion, and log-scale offsets for coordinates `means` at `times`."""
        if times.ndim == 0 or (times.ndim == 2 and times.shape[0] != means.shape[0]):
            times = times.reshape(-1, 1).expand(means.shape[0], 1)
        elif times.ndim == 1 and times.shape[0] != means.shape[0]:
            times = times.unsqueeze(-1).expand(means.shape[0], 1)
        elif times.ndim == 1:
            times = times.unsqueeze(-1)

        feat_p = self.pos_encoder(means)
        feat_t = self.time_encoder(times)
        feat_in = torch.cat([feat_p, feat_t], dim=-1)

        h = self.block1(feat_in)
        h = self.block2(torch.cat([h, feat_in], dim=-1))

        d_pos = self.delta_pos(h)
        d_rot = self.delta_rot(h)
        d_scale = self.delta_scale(h)
        return d_pos, d_rot, d_scale


class DeformableGaussianModel(nn.Module):
    """Canonical 3D Gaussians paired with a continuous deformation field."""

    def __init__(self, num_points: int = 20000, device: str = "cuda:0") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.deformation_field = DeformationMLP().to(self.device)

        # Canonical parameters
        self._means = nn.Parameter(torch.zeros(num_points, 3, device=self.device))
        quats = torch.zeros(num_points, 4, device=self.device)
        quats[:, 0] = 1.0  # Unit quaternion [w, x, y, z]
        self._quats = nn.Parameter(quats)
        self._scales = nn.Parameter(torch.full((num_points, 3), -3.9, device=self.device))
        self._opacities = nn.Parameter(torch.full((num_points, 1), 1.386, device=self.device))
        self._shs = nn.Parameter(torch.zeros(num_points, 1, 3, device=self.device))

    @property
    def num_points(self) -> int:
        return self._means.shape[0]

    def initialize_from_pcd(self, points: np.ndarray | torch.Tensor, colors: np.ndarray | torch.Tensor | None = None) -> None:
        """Initialize canonical parameters from visual-hull sparse_pcd.ply vertices."""
        if isinstance(points, np.ndarray):
            pts = torch.from_numpy(points.astype(np.float32)).to(self.device)
        else:
            pts = points.to(dtype=torch.float32, device=self.device)

        n = pts.shape[0]
        self._means = nn.Parameter(pts)
        quats = torch.zeros(n, 4, device=self.device)
        quats[:, 0] = 1.0
        self._quats = nn.Parameter(quats)
        self._scales = nn.Parameter(torch.full((n, 3), -3.9, device=self.device))
        self._opacities = nn.Parameter(torch.full((n, 1), 1.386, device=self.device))

        if colors is not None:
            if isinstance(colors, np.ndarray):
                rgb = torch.from_numpy(colors.astype(np.float32) / 255.0).to(self.device)
            else:
                rgb = (colors / 255.0).to(dtype=torch.float32, device=self.device)
            sh0 = (rgb - 0.5) / 0.28209479177387814  # Convert RGB to SH degree 0
            self._shs = nn.Parameter(sh0.unsqueeze(1))
        else:
            self._shs = nn.Parameter(torch.zeros(n, 1, 3, device=self.device))
        LOGGER.info("Initialized Deformable-GS model with %d canonical Gaussians", n)

    def get_deformed_gaussians(self, t: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deform canonical 3D Gaussians to normalized time t in [0, 1]."""
        if not isinstance(t, torch.Tensor):
            t_tensor = torch.full((self.num_points, 1), float(t), device=self.device, dtype=torch.float32)
        else:
            t_tensor = t.to(device=self.device, dtype=torch.float32)

        d_pos, d_rot, d_scale = self.deformation_field(self._means, t_tensor)

        means_t = self._means + d_pos
        quats_t = F.normalize(self._quats + d_rot, p=2, dim=-1)
        scales_t = torch.exp(self._scales + d_scale)
        opacities_t = torch.sigmoid(self._opacities)
        return means_t, scales_t, quats_t, opacities_t, self._shs
