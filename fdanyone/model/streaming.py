"""Production-grade pinned double-buffered DiT block streamer and compute cache.

Implements zero-reallocation asynchronous layer streaming from persistent pinned host RAM
to GPU VRAM and timestep modulation caching (TeaCache).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence
import torch
import torch.nn as nn

LOGGER = logging.getLogger("fdanyone.streaming")


class DiTBlockStreamer:
    """Zero-reallocation double-buffered block streamer with persistent pinned host memory."""

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        device: str | torch.device,
        *,
        pinned_memory: bool = True,
        async_streams: bool = True,
    ) -> None:
        self.blocks = list(blocks)
        self.device = torch.device(device)
        self.async_streams = async_streams and self.device.type == "cuda"
        self.pinned_memory = pinned_memory and self.device.type == "cuda"

        self.copy_stream = torch.cuda.Stream(device=self.device) if self.async_streams else None
        self._pinned_state: list[dict[str, torch.Tensor]] = []
        self._init_persistent_host_buffers()

    def _init_persistent_host_buffers(self) -> None:
        """Store initial block weights and buffers in permanently pinned host tensors."""
        for block in self.blocks:
            block_state: dict[str, torch.Tensor] = {}
            for name, tensor in list(block.named_parameters()) + list(block.named_buffers()):
                cpu_tensor = tensor.detach().to("cpu")
                if self.pinned_memory and not cpu_tensor.is_pinned():
                    try:
                        cpu_tensor = cpu_tensor.pin_memory()
                    except RuntimeError:
                        pass
                block_state[name] = cpu_tensor
            self._pinned_state.append(block_state)
            block.to("cpu")

    def _load_block_to_device(self, idx: int, non_blocking: bool = True) -> None:
        block = self.blocks[idx]
        state = self._pinned_state[idx]
        for name, param in block.named_parameters(recurse=True):
            param.data = state[name].to(self.device, non_blocking=non_blocking)
        for name, buffer in block.named_buffers(recurse=True):
            buffer.data = state[name].to(self.device, non_blocking=non_blocking)

    def _evict_block_to_cpu(self, idx: int) -> None:
        """Evict GPU tensors by replacing parameter storages with persistent pinned host references."""
        block = self.blocks[idx]
        state = self._pinned_state[idx]
        for name, param in block.named_parameters(recurse=True):
            param.data = state[name]
        for name, buffer in block.named_buffers(recurse=True):
            buffer.data = state[name]

    def forward_blocks(
        self,
        x: torch.Tensor,
        x_src: torch.Tensor | None = None,
        block_kwargs_fn: Callable[[int, torch.Tensor, torch.Tensor | None], dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Execute transformer blocks with strict bi-directional stream synchronization.

        Handles Wan2.2 tuple returns `(x, x_src)` and single-tensor returns cleanly.
        """
        num_blocks = len(self.blocks)
        if num_blocks == 0:
            return x, x_src

        compute_stream = torch.cuda.current_stream(device=self.device) if self.async_streams else None

        # Prefetch block 0
        if self.async_streams and self.copy_stream is not None:
            with torch.cuda.stream(self.copy_stream):
                self._load_block_to_device(0, non_blocking=True)
        else:
            self._load_block_to_device(0, non_blocking=False)

        for idx in range(num_blocks):
            # Compute stream must wait for prefetch to complete
            if self.async_streams and compute_stream is not None and self.copy_stream is not None:
                compute_stream.wait_stream(self.copy_stream)

            # Asynchronously prefetch next block (idx + 1) on copy_stream
            if idx + 1 < num_blocks and self.async_streams and self.copy_stream is not None:
                # Copy stream waits for compute stream before initiating next transfer
                self.copy_stream.wait_stream(compute_stream)
                with torch.cuda.stream(self.copy_stream):
                    self._load_block_to_device(idx + 1, non_blocking=True)
            elif idx + 1 < num_blocks and not self.async_streams:
                self._load_block_to_device(idx + 1, non_blocking=False)

            # Prepare block arguments
            kwargs = block_kwargs_fn(idx, x, x_src) if block_kwargs_fn is not None else {}
            
            # Execute forward pass (handles Wan2.2 tuple or single tensor signatures)
            if x_src is not None:
                out = self.blocks[idx](x, x_src, **kwargs)
            else:
                out = self.blocks[idx](x, **kwargs)

            if isinstance(out, tuple):
                x = out[0]
                if len(out) > 1:
                    x_src = out[1]
            else:
                x = out

            # Evict current block references back to host
            self._evict_block_to_cpu(idx)

        return x, x_src


class TeaCacheController:
    """Manages modulation delta tracking for timestep compute skipping (TeaCache)."""

    def __init__(self, threshold: float = 0.08) -> None:
        self.threshold = threshold
        self.previous_modulation: torch.Tensor | None = None
        self.cached_residual: torch.Tensor | None = None
        self.skipped_steps = 0
        self.total_steps = 0

    def should_skip_step(self, current_modulation: torch.Tensor) -> bool:
        """Evaluate whether to reuse the previous timestep residual."""
        if self.threshold <= 0.0 or self.previous_modulation is None:
            self.previous_modulation = current_modulation.detach().clone()
            self.total_steps += 1
            return False

        self.total_steps += 1
        delta = torch.norm(current_modulation - self.previous_modulation) / (
            torch.norm(self.previous_modulation) + 1e-6
        )

        if delta.item() < self.threshold and self.cached_residual is not None:
            self.skipped_steps += 1
            LOGGER.debug("TeaCache skipped step (delta=%.4f < %.4f)", delta.item(), self.threshold)
            return True

        self.previous_modulation = current_modulation.detach().clone()
        return False

    def save_residual(self, residual: torch.Tensor) -> None:
        """Cache the predicted residual for potential reuse."""
        self.cached_residual = residual.detach()

    def reset(self) -> None:
        """Reset state between inference runs."""
        self.previous_modulation = None
        self.cached_residual = None
        self.skipped_steps = 0
        self.total_steps = 0
