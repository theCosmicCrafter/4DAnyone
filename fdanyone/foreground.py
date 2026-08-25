"""Pinned BiRefNet inference over the canonical source clip."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from PIL import Image

from fdanyone.config import FOREGROUND


def predict_foreground_masks(
    frames: tuple[np.ndarray, ...],
    model_path: str | Path,
    device: str,
    *,
    batch_size: int = FOREGROUND.batch_size,
) -> np.ndarray:
    """Return full-raster 8-bit foreground masks for the canonical clip."""

    import torch
    from torchvision import transforms
    from torchvision.transforms.functional import to_pil_image
    from transformers import AutoModelForImageSegmentation

    if not frames:
        raise ValueError("Foreground inference requires at least one frame.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    shape = frames[0].shape
    if any(frame.dtype != np.uint8 or frame.shape != shape for frame in frames):
        raise ValueError("Foreground frames must share one RGB uint8 raster.")

    model = AutoModelForImageSegmentation.from_pretrained(
        str(Path(model_path).expanduser().resolve()),
        local_files_only=True,
        trust_remote_code=True,
    )
    model = model.eval().half().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize(FOREGROUND.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    output: list[np.ndarray] = []
    try:
        for start in range(0, len(frames), batch_size):
            images = [Image.fromarray(frame, mode="RGB") for frame in frames[start : start + batch_size]]
            inputs = torch.stack([transform(image) for image in images]).to(device=device, dtype=torch.float16)
            with torch.inference_mode():
                predictions = model(inputs)[-1].sigmoid().cpu()
            for image, prediction in zip(images, predictions, strict=True):
                mask = to_pil_image(prediction).resize(image.size).convert("L")
                output.append(np.asarray(mask, dtype=np.uint8).copy())
            del inputs, predictions
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.stack(output)


def extract_masked_videos(
    run_dir: str | Path,
    model_dir: str | Path = "models",
    backdrop: str = "green",
    device: str = "cuda:0",
) -> list[Path]:
    """Generate clean background-removed (green-screen, black, or white) videos for all generated camera angles."""
    from fdanyone.assets import resolve_foreground_model
    from fdanyone.download import ensure_foreground_model
    from fdanyone.errors import FourDAnyoneError
    from fdanyone.video import iter_rgb_video, write_video

    run_path = Path(run_dir).expanduser().resolve()
    dense_dir = run_path / "videos" / "dense"
    if not dense_dir.is_dir():
        raise FourDAnyoneError(f"No dense videos found in {dense_dir}")

    ensure_foreground_model(model_dir)
    model_path = resolve_foreground_model(model_dir)

    out_dir = run_path / "videos" / f"cutout_{backdrop}"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(list(dense_dir.glob("*.mp4")))
    results: list[Path] = []

    # Backdrop RGB colors
    color_map = {
        "green": np.array([0, 255, 0], dtype=np.float32),
        "black": np.array([0, 0, 0], dtype=np.float32),
        "white": np.array([255, 255, 255], dtype=np.float32),
    }
    bg_color = color_map.get(backdrop.lower(), color_map["green"])

    for vid_path in video_files:
        frames = tuple(iter_rgb_video(vid_path))
        if not frames:
            continue
        masks = predict_foreground_masks(frames, model_path, device=device)

        composited_frames = []
        for frame, mask in zip(frames, masks):
            alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
            # Blend frame over backdrop
            comp = (frame.astype(np.float32) * alpha + bg_color * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
            composited_frames.append(comp)

        out_path = out_dir / vid_path.name
        write_video(composited_frames, out_path, fps=25)
        results.append(out_path)

    return results

