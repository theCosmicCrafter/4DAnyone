"""Export 4DAnyone multi-view video sequences as dynamic 4D Gaussian Splatting (4DGS) datasets.

Generates `transforms_4d.json` containing camera poses and normalized timestamps (`time: t / 120.0`)
along with extracted per-frame RGB images, foreground masks, and visual-hull initialization point clouds.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Generator

import av
import numpy as np
from PIL import Image

from fdanyone.assets import resolve_foreground_model
from fdanyone.device import select_cuda_device
from fdanyone.download import ensure_foreground_model
from fdanyone.errors import FourDAnyoneError
from fdanyone.foreground import predict_foreground_masks
from fdanyone.io import AtomicResultDirectory, write_json
from fdanyone.nerfstudio.cameras import camera_to_nerfstudio
from fdanyone.nerfstudio.visual_hull import (
    NERFSTUDIO_POINT_CLOUD,
    build_sparse_point_cloud,
    write_sparse_point_cloud,
)

LOGGER = logging.getLogger("fdanyone.exporter_4d")
NERFSTUDIO_MASK_THRESHOLD = 128


def _read_cameras(result: Path) -> dict:
    path = result / "cameras.json"
    try:
        cameras = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise FourDAnyoneError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(cameras, dict):
        raise FourDAnyoneError("cameras.json must contain a JSON object.")
    return cameras


def _camera_records(rig: dict) -> list[dict]:
    if not isinstance(rig, dict) or rig.get("camera_model") != "OPENCV":
        raise FourDAnyoneError("cameras.json has no supported OPENCV camera rig.")
    cameras = rig.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise FourDAnyoneError("Camera rig must contain at least one camera.")
    return cameras


def _stream_video_frames(video_path: Path) -> Generator[np.ndarray, None, None]:
    """Stream decoded video frames one by one without full-sequence RAM buffering."""
    try:
        with av.open(str(video_path), mode="r") as container:
            streams = container.streams.video
            if not streams:
                raise FourDAnyoneError(f"No video stream in {video_path}")
            for frame in container.decode(streams[0]):
                yield frame.to_ndarray(format="rgb24")
    except OSError as exc:
        raise FourDAnyoneError(f"Cannot decode video {video_path}: {exc}") from exc


def export_4dgs_dataset(
    *,
    result_dir: str | Path,
    output_dir: str | Path,
    num_frames: int = 121,
    export_masks: bool = True,
    device: str = "cuda:0",
    model_dir: str | Path = "models",
    batch_size: int = 8,
) -> dict:
    """Export complete 4D sequence with camera trajectories, timestamps, and masks."""
    result = Path(result_dir).expanduser().resolve()
    cameras_dict = _read_cameras(result)
    camera_list = _camera_records(cameras_dict)
    num_cameras = len(camera_list)

    LOGGER.info("Exporting 4DGS dataset for %d cameras x %d frames", num_cameras, num_frames)

    foreground_model = None
    if export_masks:
        try:
            device, _ = select_cuda_device(device)
            ensure_foreground_model(model_dir)
            foreground_model = resolve_foreground_model(model_dir)
        except Exception as exc:
            LOGGER.warning("Could not initialize BiRefNet for masks (%s); exporting without masks", exc)
            export_masks = False

    atomic = AtomicResultDirectory(output_dir)
    with atomic as work_dir:
        images_dir = work_dir / "images"
        masks_dir = work_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        if export_masks:
            masks_dir.mkdir(parents=True, exist_ok=True)

        frames_meta: list[dict] = []
        canonical_frames: list[np.ndarray] = []
        canonical_masks: list[np.ndarray] = []

        with ThreadPoolExecutor(max_workers=4) as io_pool:
            for cam_meta in camera_list:
                cam_id = int(cam_meta["camera_id"])
                video_path = result / f"videos/dense/{cam_id:02d}.mp4"
                if not video_path.is_file():
                    raise FourDAnyoneError(f"Dense target video does not exist: {video_path}")

                K = np.asarray(cam_meta["K"], dtype=np.float64)
                c2w = np.asarray(cam_meta["camera_to_world"], dtype=np.float64)
                transform_matrix = camera_to_nerfstudio(c2w)
                img_w = int(cam_meta.get("image_width", 704))
                img_h = int(cam_meta.get("image_height", 1280))

                frame_gen = _stream_video_frames(video_path)
                frame_buffer: list[np.ndarray] = []
                frame_indices: list[int] = []

                for t_idx in range(num_frames):
                    raw_frame = next(frame_gen)
                    if t_idx == 0:
                        canonical_frames.append(raw_frame)
                    frame_buffer.append(raw_frame)
                    frame_indices.append(t_idx)

                    # Flush batch for segmentation & write
                    if len(frame_buffer) == batch_size or t_idx == num_frames - 1:
                        if export_masks and foreground_model is not None:
                            masks = predict_foreground_masks(
                                frame_buffer,
                                foreground_model,
                                device=device,
                                batch_size=batch_size,
                            )
                        else:
                            masks = [None] * len(frame_buffer)

                        if t_idx < batch_size and export_masks and masks[0] is not None:
                            canonical_masks.append(masks[0] >= NERFSTUDIO_MASK_THRESHOLD)

                        for f_idx, frame_data, mask_data in zip(frame_indices, frame_buffer, masks):
                            img_name = f"cam_{cam_id:02d}_frame_{f_idx:04d}.png"
                            io_pool.submit(Image.fromarray(frame_data).save, images_dir / img_name)

                            entry: dict[str, object] = {
                                "file_path": f"images/{img_name}",
                                "transform_matrix": transform_matrix,
                                "fl_x": float(K[0, 0]),
                                "fl_y": float(K[1, 1]),
                                "cx": float(K[0, 2]),
                                "cy": float(K[1, 2]),
                                "w": img_w,
                                "h": img_h,
                                "camera_id": cam_id,
                                "frame_index": f_idx,
                                "time": float(f_idx) / max(1, num_frames - 1),
                            }
                            if mask_data is not None:
                                mask_name = f"cam_{cam_id:02d}_frame_{f_idx:04d}.png"
                                bin_mask = np.where(mask_data >= NERFSTUDIO_MASK_THRESHOLD, 255, 0).astype(np.uint8)
                                io_pool.submit(Image.fromarray(bin_mask).save, masks_dir / mask_name)
                                entry["mask_path"] = f"masks/{mask_name}"

                            frames_meta.append(entry)

                        frame_buffer.clear()
                        frame_indices.clear()

        # Build visual hull point cloud from canonical frame 0
        if canonical_masks and len(canonical_masks) == len(canonical_frames):
            LOGGER.info("Building initial 3D visual-hull point cloud")
            pcd_points, pcd_colors = build_sparse_point_cloud(
                tuple(canonical_frames),
                np.stack(canonical_masks),
                camera_list,
                device,
            )
            write_sparse_point_cloud(work_dir / NERFSTUDIO_POINT_CLOUD, pcd_points, pcd_colors)

        transforms_data = {
            "camera_model": "OPENCV",
            "ply_file_path": NERFSTUDIO_POINT_CLOUD,
            "frames": frames_meta,
        }
        write_json(transforms_data, work_dir / "transforms_4d.json", sort_keys=False)
        write_json(transforms_data, work_dir / "transforms.json", sort_keys=False)

    summary = {
        "output_dir": str(output_dir),
        "num_cameras": num_cameras,
        "num_frames": num_frames,
        "total_dataset_images": len(frames_meta),
        "transforms_file": str(Path(output_dir) / "transforms_4d.json"),
    }
    LOGGER.info("4DGS dataset export complete: %d images", len(frames_meta))
    return summary
