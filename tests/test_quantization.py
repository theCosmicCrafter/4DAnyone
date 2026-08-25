"""Unit tests for Hadamard rotation and FP8 quantization helpers."""

import torch
import torch.nn as nn
from fdanyone.model.quantization import (
    get_hadamard_matrix,
    apply_hadamard_rotation,
    quantize_model_fp8,
)


def test_hadamard_matrix_orthogonal():
    h = get_hadamard_matrix(16)
    assert h.shape == (16, 16)
    # Check H @ H.T == I (orthogonality)
    ident = torch.matmul(h, h.t())
    expected = torch.eye(16)
    assert torch.allclose(ident, expected, atol=1e-5)


def test_apply_hadamard_rotation():
    w = torch.randn(32, 16)
    w_rot, h = apply_hadamard_rotation(w)
    assert w_rot.shape == (32, 16)
    assert h is not None


def test_quantize_model_fp8():
    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.fc2 = nn.Linear(32, 16)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    net = DummyNet()
    quantize_model_fp8(net)
    x = torch.randn(4, 16)
    out = net(x)
    assert out.shape == (4, 16)
