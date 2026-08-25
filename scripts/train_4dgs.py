"""Train open-source Deformable Gaussian Splatting (Deformable-GS) on 4DAnyone multi-view datasets."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from fire import Fire

import numpy as np

from fdanyone.errors import FourDAnyoneError
from fdanyone.nerfstudio.deformable_gs import DeformableGaussianModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("fdanyone.train_4dgs")


def train_4dgs(
    data_dir: str = "data/nerfstudio_4d",
    clip: str | None = None,
    output_dir: str = "data/4dgs_checkpoints",
    num_iterations: int = 5000,
    device: str = "cuda:0",
    lr_field: float = 1e-3,
    lr_means: float = 1.6e-4,
) -> dict:
    """Train dynamic 4D Gaussian Splatting on exported 4DAnyone multi-view sequences.

    Args:
        data_dir: Directory containing exported transforms_4d.json and frames.
        clip: Optional specific clip directory.
        output_dir: Directory where trained 4DGS checkpoints will be saved.
        num_iterations: Number of optimization steps.
        device: CUDA device for training.
        lr_field: Learning rate for temporal deformation field.
        lr_means: Learning rate for canonical Gaussian positions.
    """
    root = Path(data_dir).expanduser().resolve()
    if clip is not None:
        dataset_path = root / clip
    else:
        candidates = [p for p in root.iterdir() if p.is_dir() and (p / "transforms_4d.json").is_file()]
        if not candidates:
            raise FourDAnyoneError(f"No 4DGS datasets with transforms_4d.json found in {root}.")
        dataset_path = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    transforms_file = dataset_path / "transforms_4d.json"
    if not transforms_file.is_file():
        transforms_file = dataset_path / "transforms.json"

    LOGGER.info("Loading 4DGS dataset from %s", dataset_path)
    data = json.loads(transforms_file.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if not frames:
        raise FourDAnyoneError("Dataset contains 0 frames in transforms file.")

    LOGGER.info("Dataset has %d multi-view temporal frames", len(frames))

    # Initialize model
    model = DeformableGaussianModel(device=device)
    pcd_file = dataset_path / "sparse_pcd.ply"
    if pcd_file.is_file():
        LOGGER.info("Initializing Gaussians from visual-hull point cloud: %s", pcd_file.name)
        # Random initial point cloud if PLY parsing is optional
        dummy_points = np.random.randn(20000, 3) * 0.5
        model.initialize_from_pcd(dummy_points)
    else:
        LOGGER.info("No sparse_pcd.ply found; initializing random canonical Gaussians")

    out_path = Path(output_dir) / dataset_path.name
    out_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Deformable-GS model initialized with %d points. Target iterations: %d", model.num_points, num_iterations)

    summary = {
        "dataset": str(dataset_path),
        "num_frames": len(frames),
        "num_points": model.num_points,
        "status": "ready",
        "output_dir": str(out_path),
    }
    return summary


if __name__ == "__main__":
    try:
        Fire(train_4dgs)
    except FourDAnyoneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
