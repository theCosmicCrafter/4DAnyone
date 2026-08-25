"""Unit tests for Deformable Gaussian Splatting architecture."""

import torch
from fdanyone.nerfstudio.deformable_gs import (
    SinusoidalEncoder,
    DeformationMLP,
    DeformableGaussianModel,
)


def test_sinusoidal_encoding():
    enc = SinusoidalEncoder(in_channels=3, num_frequencies=4)
    x = torch.randn(10, 3)
    out = enc(x)
    # 3 * 4 * 2 = 24
    assert out.shape == (10, 24)


def test_deformation_field_forward():
    field = DeformationMLP(
        spatial_frequencies=4,
        temporal_frequencies=4,
        hidden_dim=32,
    )
    xyz = torch.randn(50, 3)
    t = torch.tensor([[0.5]])
    d_xyz, d_rot, d_scale = field(xyz, t)

    assert d_xyz.shape == (50, 3)
    assert d_rot.shape == (50, 4)
    assert d_scale.shape == (50, 3)


def test_deformable_gaussian_model():
    model = DeformableGaussianModel(num_points=100, device="cpu")
    means_t, scales_t, rots_t, opacities_t, features_t = model.get_deformed_gaussians(t=0.25)

    assert means_t.shape == (100, 3)
    assert scales_t.shape == (100, 3)
    assert rots_t.shape == (100, 4)
    assert opacities_t.shape == (100, 1)
    assert features_t.shape == (100, 1, 3)
