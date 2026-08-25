"""Export 4DAnyone multi-view video sequences to dynamic 4DGS dataset format."""

from __future__ import annotations

import sys
from pathlib import Path
from fire import Fire

from fdanyone.errors import FourDAnyoneError
from fdanyone.nerfstudio.exporter_4d import export_4dgs_dataset


def export_4d(
    result_dir: str = "data/fdanyone",
    output_dir: str = "data/nerfstudio_4d",
    clip: str | None = None,
    num_frames: int = 121,
    export_masks: bool = True,
    device: str = "cuda:0",
    model_dir: str = "models",
) -> dict:
    """Export multi-view video sequences for 4DGS reconstruction.

    Args:
        result_dir: Directory containing 4DAnyone generation outputs.
        output_dir: Directory where the 4DGS dataset will be written.
        clip: Optional specific clip subdirectory name.
        num_frames: Number of temporal frames to export (default 121).
        export_masks: Whether to generate and export BiRefNet foreground masks.
        device: CUDA device for mask generation.
        model_dir: Directory holding model weights.
    """
    root = Path(result_dir).expanduser().resolve()
    if clip is not None:
        target_dir = root / clip
    else:
        # Pick the most recent generation subdirectory
        candidates = [p for p in root.iterdir() if p.is_dir() and (p / "cameras.json").is_file()]
        if not candidates:
            raise FourDAnyoneError(f"No valid 4DAnyone results found in {root}.")
        target_dir = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    out_path = Path(output_dir) / target_dir.name
    return export_4dgs_dataset(
        result_dir=target_dir,
        output_dir=out_path,
        num_frames=num_frames,
        export_masks=export_masks,
        device=device,
        model_dir=model_dir,
    )


if __name__ == "__main__":
    try:
        Fire(export_4d)
    except FourDAnyoneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
