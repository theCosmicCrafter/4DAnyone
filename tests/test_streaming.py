"""Unit tests for DiT block streaming and TeaCache controllers."""

import torch
import torch.nn as nn
from fdanyone.model.streaming import DiTBlockStreamer, TeaCacheController


class DummyBlock(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x, **kwargs):
        return self.linear(x)


def test_dit_block_streamer_cpu():
    blocks = [DummyBlock() for _ in range(4)]
    streamer = DiTBlockStreamer(blocks, device="cpu", pinned_memory=False, async_streams=False)
    x = torch.randn(2, 32)
    out, out_src = streamer.forward_blocks(x)
    assert out.shape == (2, 32)


def test_teacache_controller():
    cache = TeaCacheController(threshold=0.1)
    mod1 = torch.ones(1, 16)
    assert not cache.should_skip_step(mod1)

    cache.save_residual(torch.ones(1, 16))

    # Slightly changed modulation (delta < 0.1) -> should skip
    mod2 = torch.ones(1, 16) * 1.02
    assert cache.should_skip_step(mod2)
    assert cache.skipped_steps == 1

    # Significantly changed modulation (delta > 0.1) -> should not skip
    mod3 = torch.ones(1, 16) * 2.0
    assert not cache.should_skip_step(mod3)
