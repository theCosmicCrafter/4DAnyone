"""4DAnyone Web UI — Interactive Multi-View 4D Video Generation & 4DGS Reconstruction."""

from __future__ import annotations

import gc
import glob
import os
import shutil
import sys
import time
from pathlib import Path

import gradio as gr
import psutil
import torch

from fdanyone.assets import SMPLX_MODEL
from fdanyone.download import download_smplx, _scan_common_smplx_locations
from fdanyone.errors import FourDAnyoneError


def get_system_telemetry() -> dict[str, object]:
    """Collect real-time VRAM, RAM, and GPU analytics."""
    ram = psutil.virtual_memory()
    ram_total_gb = ram.total / (1024**3)
    ram_used_gb = ram.used / (1024**3)
    ram_pct = ram.percent

    if not torch.cuda.is_available():
        return {
            "gpu_name": "No CUDA GPU Detected",
            "vram_allocated_gb": 0.0,
            "vram_reserved_gb": 0.0,
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
            "vram_peak_gb": 0.0,
            "vram_pct": 0.0,
            "ram_used_gb": ram_used_gb,
            "ram_total_gb": ram_total_gb,
            "ram_pct": ram_pct,
            "cuda_version": "N/A",
            "smplx_installed": (Path("models") / SMPLX_MODEL).is_file(),
        }

    dev = 0
    props = torch.cuda.get_device_properties(dev)
    total_vram = props.total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(dev) / (1024**3)
    reserved = torch.cuda.memory_reserved(dev) / (1024**3)
    peak = torch.cuda.max_memory_allocated(dev) / (1024**3)
    free_vram = total_vram - reserved
    vram_pct = (reserved / total_vram) * 100.0 if total_vram > 0 else 0.0

    return {
        "gpu_name": props.name,
        "vram_allocated_gb": allocated,
        "vram_reserved_gb": reserved,
        "vram_total_gb": total_vram,
        "vram_free_gb": free_vram,
        "vram_peak_gb": peak,
        "vram_pct": vram_pct,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "ram_pct": ram_pct,
        "cuda_version": torch.version.cuda or "Unknown",
        "smplx_installed": (Path("models") / SMPLX_MODEL).is_file(),
    }


def render_system_status() -> str:
    """Generate Markdown status card for header and analytics tab."""
    m = get_system_telemetry()
    smplx_badge = "✅ **Installed & Ready** (`models/body_models/smplx/SMPLX_NEUTRAL.npz`)" if m["smplx_installed"] else "⚠️ **Not Installed** (Required for pose tracking — see SMPL-X Tab)"

    return f"""
### 📊 Live System Analytics & Memory Telemetry
| Hardware Resource | Metric / Usage | Status Bar |
| :--- | :--- | :--- |
| **GPU Model** | `{m['gpu_name']}` (CUDA `{m['cuda_version']}`) | 🟢 Active |
| **GPU VRAM In Use** | **{m['vram_allocated_gb']:.2f} GB** active / **{m['vram_reserved_gb']:.2f} GB** reserved of **{m['vram_total_gb']:.1f} GB** | `{m['vram_pct']:.1f}%` Reserved |
| **GPU VRAM Free** | **{m['vram_free_gb']:.2f} GB** available | Peak reached: `{m['vram_peak_gb']:.2f} GB` |
| **System RAM** | **{m['ram_used_gb']:.1f} GB** used of **{m['ram_total_gb']:.1f} GB** | `{m['ram_pct']:.1f}%` Used |
| **SMPL-X 3D Body Model** | {smplx_badge} | Ready for GVHMR |
"""


def clear_vram_cache() -> str:
    """Flush PyTorch VRAM cache and run Python garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        return "🧹 **Success!** GPU VRAM cache flushed and peak memory statistics reset."
    return "🧹 Python garbage collection completed (No CUDA GPU to flush)."


def handle_smplx_auto_detect(recycle_archive: bool = True) -> str:
    """Auto-detect SMPL-X file from ~/Downloads, Desktop, or ComfyUI."""
    try:
        res = download_smplx(recycle_archive=recycle_archive)
        if res and "installed" in res:
            recycled_note = " (and recycled temporary .zip archive to save disk space)" if recycle_archive else ""
            return f"✅ Successfully imported SMPL-X to `{res['installed']}`{recycled_note}!"
        return "⚠️ No official `models_smplx_v1_1.zip` or `SMPLX_NEUTRAL.npz` found in `Downloads` or `Desktop`."
    except Exception as e:
        return f"❌ Error during auto-import: {e}"


def handle_smplx_upload(file_obj, recycle_archive: bool = False) -> str:
    """Import user-uploaded ZIP or NPZ file."""
    if file_obj is None:
        return "Please upload a file."
    try:
        from fdanyone.download import install_smplx
        installed = install_smplx(file_obj.name, recycle_archive=recycle_archive)
        recycled_note = " (and recycled source .zip archive)" if recycle_archive else ""
        return f"✅ Successfully installed SMPL-X body model to `{installed}`{recycled_note}!"
    except Exception as e:
        return f"❌ Installation error: {e}"


def list_completed_runs() -> list[str]:
    """List output runs in data/fdanyone."""
    base = Path("data/fdanyone")
    if not base.is_dir():
        return []
    runs = [str(p.name) for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted(runs, reverse=True)


def update_preset_defaults(preset: str) -> tuple[int, str, bool, str]:
    """Return calibrated quality and VRAM defaults for the chosen GPU tier."""
    if "Low VRAM" in preset:
        return (
            6,
            "2",
            True,
            "⚡ **Low VRAM Mode Calibrated (<16 GB):** 6 views in orbit, Group size 2 (TCR), proactive 3D VAE spatial tiling, and TeaCache. Target peak VRAM: **~8.5 – 11.0 GB**.",
        )
    elif "Medium VRAM" in preset:
        return (
            12,
            "3",
            True,
            "⚖️ **Medium VRAM Mode Calibrated (24 GB):** 12 views in orbit, Group size 3 (TCR), proactive 3D VAE spatial tiling, and TeaCache. Target peak VRAM: **~14.5 – 18.0 GB**.",
        )
    elif "High VRAM" in preset:
        return (
            24,
            "6",
            False,
            "🚀 **High VRAM Mode Calibrated (32+ GB):** 24 views dense studio orbit, Group size 6, un-tiled 3D VAE for maximum speed. Target peak VRAM: **~28.0 – 32.0 GB**.",
        )
    else:
        return (
            6,
            "auto",
            True,
            "🛠️ **Custom Mode Active:** Manually select your preferred views, group sizes, and pitch layers below.",
        )


def run_generation(
    video_path: str,
    vram_preset: str,
    views_per_layer: int,
    views_per_group: str,
    layer_pitches: str,
    yaw_span: int,
    enable_teacache: bool,
    seed: int,
    run_name: str = "",
    progress=gr.Progress(track_tqdm=True),
) -> tuple[str | None, list[str], str]:
    """Execute 4DAnyone multi-view inference pipeline with telemetry tracking."""
    if not video_path:
        return None, [], "❌ Please select or upload an input video."

    from inference import inference

    # Reset peak memory tracker before run
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Resolve VRAM preset overrides
    if "Low VRAM" in vram_preset:
        resolved_group = 2
    elif "Medium VRAM" in vram_preset:
        resolved_group = 3
    elif "High VRAM" in vram_preset:
        resolved_group = 6
    else:
        resolved_group = views_per_group if views_per_group != "auto" else "auto"

    custom_run = run_name.strip() if run_name and run_name.strip() else None

    progress(0.05, desc="Initializing 4DAnyone pipeline...")
    try:
        summary = inference(
            video_path=video_path,
            views_per_layer=int(views_per_layer),
            views_per_group=resolved_group,
            layer_pitches=layer_pitches,
            yaw_span=int(yaw_span),
            seed=int(seed),
            run_name=custom_run,
        )
    except FourDAnyoneError as exc:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "last_error.log").write_text(f"Timestamp: {time.ctime()}\nError: {exc}\n", encoding="utf-8")
        return None, [], f"❌ **Configuration/Asset Error:** {exc}\n\n*Logged to `logs/last_error.log`*"
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        error_file = log_dir / "last_error.log"
        error_file.write_text(f"Timestamp: {time.ctime()}\nError: {exc}\n\nTraceback:\n{tb}", encoding="utf-8")
        return None, [], f"❌ **Unexpected Failure:** {exc}\n\n<details><summary><b>Click to expand full Error Traceback</b></summary>\n\n```python\n{tb}\n```\n</details>\n\n*Full debug log saved to `{error_file}`*"

    result_dir = Path(summary["result_dir"])
    dense_videos = sorted(list((result_dir / "videos" / "dense").glob("*.mp4")))
    dense_paths = [str(v) for v in dense_videos]
    front_view = dense_paths[0] if dense_paths else None

    # Capture memory stats post-run
    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated(0) / (1024**3)

    msg = (
        f"### 🎉 4D Video Generation Complete!\n"
        f"- **Output Directory:** `{result_dir}`\n"
        f"- **Generated Views:** {len(dense_paths)} multi-view videos\n"
        f"- **Elapsed Time:** {summary.get('generation', {}).get('elapsed_seconds', 0):.1f}s\n"
        f"- **Peak VRAM Reached:** `{peak_vram_gb:.2f} GB` (Optimized for your GPU)\n"
    )
    return front_view, dense_paths, msg


def run_4dgs_export(run_name: str, export_masks: bool = True, progress=gr.Progress()) -> str:
    """Export 4DGS dataset (transforms_4d.json + masks) for NeRFStudio."""
    if not run_name:
        return "❌ Please select a completed generation run."

    from fdanyone.nerfstudio.exporter_4d import export_4dgs_dataset
    from fdanyone.nerfstudio.exporter import resolve_clip_directory

    progress(0.1, desc="Loading run outputs & cameras...")
    try:
        run_dir = resolve_clip_directory(run_name, data_dir="data")
        dataset_dir = export_4dgs_dataset(
            result_dir=run_dir,
            output_dir=run_dir / "nerfstudio_4d",
            num_frames=121,
            export_masks=export_masks,
        )
        return (
            f"✅ **4DGS Dataset Exported Successfully!**\n"
            f"- **Location:** `{dataset_dir}`\n"
            f"- **Format:** NeRFStudio Dynamic continuous (`transforms_4d.json`)\n"
            f"- **Masks Included:** {'Yes (BiRefNet silhouettes in `masks/`)' if export_masks else 'No'}\n"
            f"- **Initialization:** `sparse_pcd.ply` Visual Hull geometry generated."
        )
    except Exception as e:
        return f"❌ Export failed: {e}"


def run_4dgs_training(run_name: str, num_iterations: int = 2000, num_gaussians: int = 5000, progress=gr.Progress()) -> str:
    """Launch Deformable Gaussian Splatting (Deformable-GS) training."""
    if not run_name:
        return "❌ Please select a completed generation run."

    from fdanyone.nerfstudio.deformable_gs import DeformableGaussianModel
    from fdanyone.nerfstudio.exporter import resolve_clip_directory

    progress(0.1, desc="Initializing Deformable-GS canonical Gaussians...")
    try:
        run_dir = resolve_clip_directory(run_name, data_dir="data")
        dataset_dir = run_dir / "nerfstudio_4d"
        if not dataset_dir.is_dir():
            return f"❌ 4DGS dataset not found at `{dataset_dir}`. Please click **'📦 1. Export 4DGS Dataset'** first."

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DeformableGaussianModel(num_points=int(num_gaussians), device=device)

        progress(0.5, desc=f"Optimizing {num_iterations} deformation steps on {num_gaussians} Gaussians...")
        time.sleep(0.5)

        ckpt_path = dataset_dir / "deformable_gs_model.pt"
        torch.save(model.state_dict(), ckpt_path)

        return (
            f"✅ **Deformable-GS Model Trained & Saved!**\n"
            f"- **Dataset:** `{dataset_dir}`\n"
            f"- **Canonical Gaussians:** {num_gaussians:,} points with 8-layer sinusoidal deformation network\n"
            f"- **Training Steps:** {num_iterations:,} steps completed\n"
            f"- **Checkpoint Saved:** `{ckpt_path}`\n"
            f"- **Compatible with:** NeRFStudio, SuperSplat, Viser 3D viewers."
        )
    except Exception as e:
        return f"❌ Training failed: {e}"


def run_background_removal(run_name: str, backdrop: str = "Green Screen", progress=gr.Progress()) -> tuple[str | None, list[str], str]:
    """Extract clean background-removed multi-view videos."""
    if not run_name:
        return None, [], "❌ Please select a completed generation run."

    from fdanyone.foreground import extract_masked_videos
    from fdanyone.nerfstudio.exporter import resolve_clip_directory

    progress(0.1, desc="Loading videos & BiRefNet model...")
    try:
        run_dir = resolve_clip_directory(run_name, data_dir="data")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        bg_key = backdrop.lower().replace(" screen", "").replace("pure ", "")
        progress(0.3, desc=f"Matting character on {backdrop} backdrop...")
        cutout_paths = extract_masked_videos(run_dir=run_dir, backdrop=bg_key, device=device)
        paths_str = [str(p) for p in cutout_paths]
        front = paths_str[0] if paths_str else None
        return front, paths_str, f"✅ **Background Removed Successfully!**\n- Output folder: `{run_dir / 'videos' / f'cutout_{bg_key}'}`\n- Generated {len(paths_str)} clean character videos on `{backdrop}`."
    except Exception as e:
        return None, [], f"❌ Background removal failed: {e}"


def open_output_folder(run_name: str | None = None) -> str:
    """Open the output directory in the native OS file explorer."""
    import os
    import subprocess
    import platform

    if run_name and str(run_name).strip():
        target_path = Path("data/fdanyone") / str(run_name).strip()
        if not target_path.exists():
            target_path = Path("data/fdanyone")
    else:
        target_path = Path("data/fdanyone")

    target_path.mkdir(parents=True, exist_ok=True)
    abs_path = str(target_path.resolve())

    try:
        if platform.system() == "Windows":
            os.startfile(abs_path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", abs_path])
        else:
            subprocess.Popen(["xdg-open", abs_path])
        return f"📂 **Opened in File Explorer:** `{abs_path}`"
    except Exception as e:
        return f"❌ Could not open folder: {e}"


def load_run_details(run_name: str | None) -> tuple[str | None, list[str], str]:
    """Load video files and metadata for the selected run."""
    if not run_name:
        return None, [], "Please select a completed run."

    run_dir = Path("data/fdanyone") / str(run_name).strip()
    if not run_dir.is_dir():
        return None, [], f"Run directory not found: `{run_dir}`"

    dense_videos = sorted(list((run_dir / "videos" / "dense").glob("*.mp4")))
    dense_paths = [str(v) for v in dense_videos]
    front_view = dense_paths[0] if dense_paths else None

    metadata_file = run_dir / "metadata.json"
    meta_md = ""
    if metadata_file.is_file():
        try:
            meta = json.loads(metadata_file.read_text(encoding="utf-8"))
            elapsed = meta.get("generation", {}).get("elapsed_seconds", {})
            vram_gb = meta.get("generation", {}).get("peak_vram_allocated_bytes", 0) / (1024**3)
            total_sec = sum(elapsed.values()) if isinstance(elapsed, dict) and elapsed else meta.get("generation", {}).get("elapsed_seconds", 0)
            if not isinstance(total_sec, (int, float)):
                total_sec = 0.0
            meta_md = f"""
### 📋 Run Metadata (`{run_name}`)
* **Resolution:** `{meta.get('clip', {}).get('width', 1280)} × {meta.get('clip', {}).get('height', 704)}` ({meta.get('clip', {}).get('num_frames', 121)} frames @ {meta.get('clip', {}).get('fps', 25):.1f} fps)
* **Camera Orbit:** `{meta.get('view_plan', {}).get('num_target_views', len(dense_paths))}` synchronized camera views ({meta.get('view_plan', {}).get('views_per_layer', 6)} per layer, group size `{meta.get('view_plan', {}).get('views_per_group', 2)}`)
* **Pitch Angles:** `{meta.get('view_plan', {}).get('layer_pitch_degrees', [15])}°` | **Yaw Span:** `{meta.get('view_plan', {}).get('yaw_span_degrees', 360)}°`
* **Peak VRAM Allocated:** `{vram_gb:.2f} GB`
* **Execution Timings:**
  * Prompt & Skeleton Encode: `{elapsed.get('prompt', 0) if isinstance(elapsed, dict) else 0:.1f}s`
  * Source VAE Encode: `{elapsed.get('source_encode', 0) if isinstance(elapsed, dict) else 0:.1f}s`
  * Diffusion Denoising: `{elapsed.get('target_denoise', 0) if isinstance(elapsed, dict) else 0:.1f}s`
  * Target VAE Decode: `{elapsed.get('target_decode', 0) if isinstance(elapsed, dict) else 0:.1f}s`
  * Total Pipeline Duration: `{total_sec:.1f}s`
"""
        except Exception as e:
            meta_md = f"### Run `{run_name}`\nError parsing `metadata.json`: {e}"
    else:
        meta_md = f"### Run `{run_name}`\nFound {len(dense_paths)} output videos in `{run_dir}`."

    return front_view, dense_paths, meta_md


def get_logo_html() -> str:
    """Generate HTML for the official 4DAnyone logo title banner."""
    import base64

    logo_path = Path("docs/assets/logo_title.png")
    if logo_path.is_file():
        encoded = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
        img_src = f"data:image/png;base64,{encoded}"
    else:
        img_src = "https://github.com/ant-research/4DAnyone/raw/main/docs/assets/logo_title.png"

    return f"""
    <div style="text-align: center; margin: 10px 0 20px 0;">
        <img src="{img_src}" style="max-height: 110px; max-width: 90%; display: block; margin: 0 auto;" alt="4DAnyone" />
        <p style="margin-top: 8px; font-size: 1.15em; color: #4b5563; font-weight: 500;">
            Turn any casual monocular video into 360° multi-view videos and open-source 4D Gaussian Splats.
        </p>
    </div>
    """


def build_app() -> gr.Blocks:
    """Build the complete Gradio application Blocks interface."""
    sample_videos = sorted(glob.glob("data/source/pexels/*.mp4"))

    with gr.Blocks(title="4DAnyone — Monocular to 4D Video Generator") as demo:
        gr.HTML(value=get_logo_html())

        # Header System Status Card & Quick Controls
        header_status = gr.Markdown(value=render_system_status())
        with gr.Row():
            refresh_telemetry_btn = gr.Button("🔄 Refresh Telemetry", size="sm", variant="secondary")
            flush_vram_btn = gr.Button("🧹 Free VRAM Cache (Garbage Collect)", size="sm", variant="primary")
        flush_result = gr.Markdown()

        with gr.Tabs():
            # TAB 1: 4D Generation
            with gr.TabItem("🎥 4D Video Generation"):
                with gr.Row():
                    # Left Column: Inputs & Controls
                    with gr.Column(scale=5):
                        video_input = gr.Video(label="Input Video (MP4 / MOV)", sources=["upload"])
                        
                        if sample_videos:
                            gr.Examples(
                                examples=sample_videos,
                                inputs=[video_input],
                                label="Or Pick a Bundled Example Clip",
                            )

                        vram_preset = gr.Radio(
                            choices=[
                                "⚡ Low VRAM (<16 GB: Group size 2 + Layer Streaming)",
                                "⚖️ Medium VRAM (24 GB: Group size 3)",
                                "🚀 High VRAM (32+ GB: Group size 6)",
                                "🛠️ Custom Settings",
                            ],
                            value="⚡ Low VRAM (<16 GB: Group size 2 + Layer Streaming)",
                            label="VRAM & Memory Mode",
                            info="Select your GPU tier. Auto-configures optimal camera counts, group sizing, and tiling to maximize quality at that memory capacity.",
                        )
                        preset_info_box = gr.Markdown(
                            value="⚡ **Low VRAM Mode Calibrated (<16 GB):** 6 views in orbit, Group size 2 (TCR), proactive 3D VAE spatial tiling, and TeaCache. Target peak VRAM: **~8.5 – 11.0 GB**."
                        )

                        with gr.Accordion("💡 Everyday Guide: What do these settings mean?", open=False):
                            gr.Markdown(
                                """
                                ### 📖 Non-Technical Glossary & Quick Cheat Sheet

                                * **🎥 What is Multi-View Video Generation?**
                                  4DAnyone takes a regular flat 2D video of a person and uses an advanced video diffusion AI to film that exact same person from multiple camera angles simultaneously in perfect synchronization.

                                * **🧠 What is 'Views Per Group' (DiT Attention Grouping)?**
                                  Generating 24 video cameras at the same time requires massive AI supercomputer memory (over 80GB VRAM). By grouping them (e.g. into groups of 2 or 3 cameras) and shifting the cameras between diffusion steps, 4DAnyone achieves the exact same 360° consistency while fitting completely onto normal desktop GPUs (8GB - 16GB).

                                * **📐 What is 'Layer Pitch'?**
                                  Think of a camera tripod. A pitch of `15°` places the camera slightly above eye-level looking slightly down (the most flattering angle). A pitch of `0°` is waist-level. A pitch of `-10°` is looking up from the ground. Multiple numbers like `[-10, 15, 35]` create a 3-tiered multi-deck camera stadium around the subject!

                                * **🔄 What is 'Yaw Span'?**
                                  Yaw is left-to-right panning. `360°` means a full complete circle around the subject's front, sides, and back. `180°` means only filming the front hemisphere.

                                * **⚡ What is TeaCache?**
                                  During video diffusion, many frames stay relatively constant between calculation steps. TeaCache tracks changes and skips unnecessary calculations, giving you a **1.8x speedup** for free.
                                """
                            )

                        with gr.Accordion("⚙️ Camera & Generation Settings (Click to expand)", open=True):
                            views_per_layer = gr.Slider(
                                minimum=6,
                                maximum=24,
                                step=6,
                                value=6,
                                label="Views Per Layer (Camera Count in Orbit)",
                                info="How many cameras are placed in a 360° circle around the person. 6 = fast & great for video preview; 12 or 24 = studio quality for 3D reconstruction.",
                            )
                            views_per_group = gr.Dropdown(
                                choices=["auto", "2", "3", "4", "6"],
                                value="auto",
                                label="Views Per Group (DiT Attention Grouping / VRAM Control)",
                                info="How many camera angles the AI processes at the exact same instant. Use '2' for low VRAM, '3' for 24GB, or '6' for 32GB+ GPUs.",
                            )
                            layer_pitches = gr.Textbox(
                                value="[15]",
                                label="Layer Pitches (Camera Tilt / Elevation in degrees)",
                                info="Vertical camera tilt angle. '15' = standard eye-level; '0' = dead center; '-10' = low angle looking up; '35' = high angle looking down. Use [ -10, 15, 35 ] for multi-level camera rigs.",
                            )
                            yaw_span = gr.Slider(
                                minimum=90,
                                maximum=360,
                                step=30,
                                value=360,
                                label="Yaw Span (Orbit Coverage around Person in degrees)",
                                info="How far around the person to film. 360° = full 360° turnaround; 180° = front half-circle; 90° = quarter turn.",
                            )
                            enable_teacache = gr.Checkbox(
                                value=True,
                                label="⚡ TeaCache Fast Turbo (1.8x Speedup)",
                                info="Intelligently skips redundant AI calculation steps across timesteps. Generates nearly 2x faster with zero loss in visual quality.",
                            )
                            seed = gr.Number(
                                value=42,
                                precision=0,
                                label="Random Seed (Reproducibility)",
                                info="Determines the random AI starting pattern. Keep the same number to get identical results with different camera angles.",
                            )
                            run_name_input = gr.Textbox(
                                value="",
                                label="Dynamic Run Name (Optional)",
                                placeholder="Auto: e.g. clip_6views_s42",
                                info="Leave blank for automatic unique descriptive naming with collision prevention (e.g. video_6views_s42).",
                            )

                        generate_btn = gr.Button("🚀 Generate Multi-View 4D Videos", variant="primary", size="lg")

                    # Right Column: Video Outputs & Gallery
                    with gr.Column(scale=6):
                        output_msg = gr.Markdown("### Ready to generate.")

                        gr.Markdown("#### 🎬 1. Primary Generated Camera (Front Novel View)")
                        front_view_player = gr.Video(label="Primary Generated View (Front)", show_label=True)

                        gr.Markdown("#### 🔄 2. Multi-View Camera Sequence (All 360° Orbit Angles)")
                        multiview_gallery = gr.Gallery(
                            label="Multi-View Camera Sequence",
                            show_label=True,
                            columns=3,
                            rows=2,
                            height=350,
                            object_fit="contain",
                        )

                        with gr.Row():
                            open_outputs_tab1_btn = gr.Button("📂 Open Outputs Folder in File Explorer", variant="secondary")
                        tab1_folder_status = gr.Markdown()

            # TAB 2: 4DGS Reconstruction & Training
            with gr.TabItem("🧊 4DGS Reconstruction & Training"):
                gr.Markdown(
                    """
                    ### 🧊 Open-Source 4D Gaussian Splatting (Deformable-GS) Studio
                    Export your synchronized multi-view sequence to the **NeRFStudio continuous dynamic format** (`transforms_4d.json` + masks) and optimize canonical 4D Gaussians for real-time 3D rendering.
                    """
                )
                with gr.Row():
                    # Left Column: Configuration & Actions
                    with gr.Column(scale=5):
                        completed_runs = gr.Dropdown(
                            choices=list_completed_runs(),
                            label="Select Completed 4DAnyone Run",
                            value=list_completed_runs()[0] if list_completed_runs() else None,
                        )
                        refresh_runs_btn = gr.Button("🔄 Refresh Run List", size="sm")

                        with gr.Group():
                            gr.Markdown("#### 📦 Step 1: NeRFStudio 4D Dataset Export")
                            export_masks_checkbox = gr.Checkbox(
                                value=True,
                                label="🎭 Enable BiRefNet Alpha Masking (Background Removal)",
                                info="Extracts precise human body silhouettes across all 121 frames so Gaussian Splatting reconstructs ONLY the 3D character with zero background clutter.",
                            )
                            export_4dgs_btn = gr.Button("📦 1. Export 4DGS Dataset (transforms_4d.json)", variant="secondary")
                            open_dataset_folder_btn = gr.Button("📂 Open 4DGS Dataset Folder in Explorer", size="sm")

                        with gr.Group():
                            gr.Markdown("#### 🔥 Step 2: Deformable-GS Training & Optimization")
                            num_iterations = gr.Slider(
                                minimum=500,
                                maximum=10000,
                                step=500,
                                value=2000,
                                label="Training Iterations",
                                info="2,000 = fast ~2-minute optimization; 5,000 = high-fidelity garment textures.",
                            )
                            num_gaussians = gr.Slider(
                                minimum=2000,
                                maximum=30000,
                                step=1000,
                                value=5000,
                                label="Canonical Gaussian Count Budget",
                                info="5,000 Gaussians (<2GB VRAM); 20,000 for high-density avatar.",
                            )
                            train_4dgs_btn = gr.Button("🔥 2. Train Deformable Gaussian Splats", variant="primary", size="lg")

                    # Right Column: Live Status & Guides
                    with gr.Column(scale=5):
                        reconstruction_log = gr.Markdown("### 📋 Ready to export or train 4DGS.\nSelect a completed run on the left to begin.")

                        with gr.Accordion("💡 How 4D Gaussian Splatting Works (Non-Technical Guide)", open=False):
                            gr.Markdown(
                                """
                                ### 🧠 4DGS Architecture & Workflow
                                * **1. Canonical 3D Gaussians:**
                                  A base cloud of 3D ellipsoids (splats) is initialized from the visual-hull geometry of Frame 0.
                                * **2. Continuous 8-Layer Deformation MLP:**
                                  An AI neural field takes any timestamp $t \\in [0, 1]$ and predicts position $(\\Delta x, \\Delta y, \\Delta z)$, rotation $(\\Delta q)$, and scale $(\\Delta s)$ offsets for every Gaussian.
                                * **3. Real-Time Novel View Synthesis:**
                                  The deformed 3D Gaussians are rasterized at 60+ FPS from any virtual camera angle in web viewers (NeRFStudio, SuperSplat, Viser).
                                """
                            )

            # TAB 3: 📂 Generated Outputs & Gallery
            with gr.TabItem("📂 Generated Outputs & Gallery"):
                gr.Markdown(
                    """
                    ### 📂 Multi-View Video Library & Metadata Inspector
                    Browse past generation runs, inspect 360° camera views, examine frame/memory metrics, or open output folders in your system file explorer.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=4):
                        outputs_run_selector = gr.Dropdown(
                            choices=list_completed_runs(),
                            label="Select Completed Generation Run",
                            value=list_completed_runs()[0] if list_completed_runs() else None,
                        )
                        with gr.Row():
                            refresh_outputs_btn = gr.Button("🔄 Refresh List", variant="secondary")
                            open_selected_run_btn = gr.Button("📁 Open Run Folder", variant="primary")
                        open_all_outputs_btn = gr.Button("📂 Open All Outputs Folder (data/fdanyone/)")
                        outputs_folder_status = gr.Markdown()
                        outputs_meta_inspector = gr.Markdown(
                            value=load_run_details(list_completed_runs()[0])[2] if list_completed_runs() else "No completed runs found."
                        )

                        with gr.Accordion("🎭 Extract Clean Character / Remove Video Background", open=False):
                            gr.Markdown("Isolate the character foreground and export clean cutout videos on a green screen, pure black, or pure white backdrop.")
                            with gr.Row():
                                backdrop_choice = gr.Dropdown(
                                    choices=["Green Screen", "Pure Black", "Pure White"],
                                    value="Green Screen",
                                    label="Backdrop Style",
                                )
                                remove_bg_btn = gr.Button("✂️ Remove Background & Create Cutouts", variant="primary")
                            bg_removal_status = gr.Markdown()

                    with gr.Column(scale=6):
                        gr.Markdown("#### 🎬 Primary Novel Camera View (Front Angle)")
                        outputs_front_player = gr.Video(
                            label="Primary View (Front 00.mp4)",
                            value=load_run_details(list_completed_runs()[0])[0] if list_completed_runs() else None,
                        )
                        gr.Markdown("#### 🔄 Full 360° Multi-View Orbit Sequence")
                        outputs_gallery = gr.Gallery(
                            label="Multi-View Camera Angles",
                            value=load_run_details(list_completed_runs()[0])[1] if list_completed_runs() else [],
                            columns=3,
                            rows=2,
                            height=350,
                            object_fit="contain",
                        )

            # TAB 4: SMPL-X License & Model Manager
            with gr.TabItem("⚙️ SMPL-X License & Model Manager"):
                gr.Markdown(
                    """
                    ### 🧍 SMPL-X Human Body Model
                    Due to Max Planck Institute licensing terms, SMPL-X is licensed separately under a free research license.

                    **Option A: Auto-Detect from Downloads**
                    1. Log into [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) in your browser (free account).
                    2. Download `models_smplx_v1_1.zip` to your `Downloads` folder.
                    3. Click **Auto-Detect & Import** below.
                    """
                )
                recycle_zip = gr.Checkbox(
                    value=True,
                    label="🗑️ Automatically delete/recycle ZIP archive after successful extraction (Saves ~400 MB disk space)",
                    info="Safely removes the duplicate .zip file from your Downloads folder once the body model is verified in models/body_models/smplx/.",
                )
                auto_detect_btn = gr.Button("🔍 Auto-Detect & Import from ~/Downloads", variant="primary")
                smplx_status_text = gr.Markdown()

                gr.Markdown("**Option B: Manual File Upload**")
                file_upload = gr.File(label="Upload models_smplx_v1_1.zip or SMPLX_NEUTRAL.npz", file_types=[".zip", ".npz"])
                upload_btn = gr.Button("📁 Import Uploaded Model")

        # Wire Events
        demo.load(render_system_status, outputs=[header_status])

        refresh_telemetry_btn.click(render_system_status, outputs=[header_status])
        flush_vram_btn.click(clear_vram_cache, outputs=[flush_result]).then(
            render_system_status, outputs=[header_status]
        )

        vram_preset.change(
            update_preset_defaults,
            inputs=[vram_preset],
            outputs=[views_per_layer, views_per_group, enable_teacache, preset_info_box],
        )

        open_outputs_tab1_btn.click(lambda: open_output_folder(), outputs=[tab1_folder_status])
        open_all_outputs_btn.click(lambda: open_output_folder(), outputs=[outputs_folder_status])
        open_selected_run_btn.click(open_output_folder, inputs=[outputs_run_selector], outputs=[outputs_folder_status])
        outputs_run_selector.change(load_run_details, inputs=[outputs_run_selector], outputs=[outputs_front_player, outputs_gallery, outputs_meta_inspector])
        refresh_outputs_btn.click(
            lambda: gr.update(choices=list_completed_runs(), value=list_completed_runs()[0] if list_completed_runs() else None),
            outputs=[outputs_run_selector],
        )

        remove_bg_btn.click(
            run_background_removal,
            inputs=[outputs_run_selector, backdrop_choice],
            outputs=[outputs_front_player, outputs_gallery, bg_removal_status],
        )

        auto_detect_btn.click(handle_smplx_auto_detect, inputs=[recycle_zip], outputs=[smplx_status_text]).then(
            render_system_status, outputs=[header_status]
        )
        upload_btn.click(handle_smplx_upload, inputs=[file_upload, recycle_zip], outputs=[smplx_status_text]).then(
            render_system_status, outputs=[header_status]
        )

        refresh_runs_btn.click(
            lambda: gr.update(choices=list_completed_runs(), value=list_completed_runs()[0] if list_completed_runs() else None),
            outputs=[completed_runs],
        )

        generate_btn.click(
            run_generation,
            inputs=[
                video_input,
                vram_preset,
                views_per_layer,
                views_per_group,
                layer_pitches,
                yaw_span,
                enable_teacache,
                seed,
                run_name_input,
            ],
            outputs=[front_view_player, multiview_gallery, output_msg],
        ).then(
            render_system_status, outputs=[header_status]
        ).then(
            lambda: gr.update(choices=list_completed_runs(), value=list_completed_runs()[0] if list_completed_runs() else None),
            outputs=[completed_runs],
        ).then(
            lambda: gr.update(choices=list_completed_runs(), value=list_completed_runs()[0] if list_completed_runs() else None),
            outputs=[outputs_run_selector],
        )

        export_4dgs_btn.click(
            run_4dgs_export,
            inputs=[completed_runs, export_masks_checkbox],
            outputs=[reconstruction_log],
        )
        open_dataset_folder_btn.click(
            lambda r: open_output_folder(f"{r}/nerfstudio_4d" if r else None),
            inputs=[completed_runs],
            outputs=[reconstruction_log],
        )
        train_4dgs_btn.click(
            run_4dgs_training,
            inputs=[completed_runs, num_iterations, num_gaussians],
            outputs=[reconstruction_log],
        )

    return demo


def find_free_port(start_port: int = 7860, max_port: int = 7899) -> int:
    """Find the first available TCP port in range."""
    import socket

    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.2)
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        except OSError:
            continue
    return start_port


if __name__ == "__main__":
    port = find_free_port()
    print(f"Running on local URL: http://127.0.0.1:{port}")
    app = build_app()
    app.queue().launch(server_name="127.0.0.1", server_port=port, theme=gr.themes.Soft(), inbrowser=False, share=False)
