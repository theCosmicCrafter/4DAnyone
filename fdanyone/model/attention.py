"""Attention backend dispatcher for 4DAnyone multi-view attention layers.

Supports PyTorch SDPA, SageAttention (INT8/FP8 matrix multiply), FlashAttention-2/3,
and sparse cross-attention for background token skipping.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

LOGGER = logging.getLogger("fdanyone.attention")


def dispatch_attention(
    q: Any,
    k: Any,
    v: Any,
    *,
    backend: str = "auto",
    mask: Optional[Any] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> Any:
    """Dispatches multi-head attention to the optimal available kernel.

    Args:
        q: Query tensor [B, H, S_q, D]
        k: Key tensor [B, H, S_k, D]
        v: Value tensor [B, H, S_v, D]
        backend: "auto" | "sdpa" | "sage" | "flash"
        mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether attention is causal
        scale: Scaling factor (defaults to 1 / sqrt(D))
    """
    import torch

    backend = backend.lower()

    if backend in ("auto", "sage"):
        try:
            from sageattention import sageattn

            # SageAttention requires [B, H, S, D] format and CUDA tensors
            if q.is_cuda and mask is None:
                return sageattn(q, k, v, is_causal=is_causal, sm_scale=scale)
        except ImportError:
            if backend == "sage":
                LOGGER.warning("SageAttention requested but package not installed; falling back to SDPA")

    if backend in ("auto", "flash"):
        try:
            from flash_attn import flash_attn_func

            # FlashAttention expects [B, S, H, D]
            if q.is_cuda and mask is None:
                q_t = q.transpose(1, 2)
                k_t = k.transpose(1, 2)
                v_t = v.transpose(1, 2)
                out = flash_attn_func(q_t, k_t, v_t, dropout_p=dropout_p, causal=is_causal, softmax_scale=scale)
                return out.transpose(1, 2)
        except ImportError:
            if backend == "flash":
                LOGGER.warning("FlashAttention requested but package not installed; falling back to SDPA")

    # Default robust PyTorch SDPA (uses FlashAttention/cuDNN internally when available)
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
    )
