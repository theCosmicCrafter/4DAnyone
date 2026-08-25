<p align="center"><a href="https://4danyone.github.io/"><img src="docs/assets/logo_title.png" width="300" alt="4DAnyone"></a></p>

<h2 align="center">4DAnyone (Enhanced Edition): Create Anyone in 4D from Casual Monocular Video</h2>

<p align="center"><a href="https://4danyone.github.io/"><strong>Project Page</strong></a> &nbsp;|&nbsp; <a href="https://arxiv.org/abs/2608.20335"><strong>Original Paper</strong></a> &nbsp;|&nbsp; <a href="https://github.com/theCosmicCrafter/4DAnyone"><strong>GitHub Repository</strong></a></p>

<p align="center"><img src="docs/assets/teaser.gif" width="100%" alt="4DAnyone teaser"></p>

> [!NOTE]
> **About this Enhanced Edition:**
> The original upstream [4DAnyone](https://github.com/ant-research/4DAnyone) release was a headless, research-only command-line tool that required enterprise hardware (>32 GB VRAM) and manual terminal scripts. 
> 
> This repository transforms 4DAnyone into a complete, creator-ready production platform featuring an **interactive 4-tab Gradio Web UI**, a **1-click Pinokio app launcher**, **low-VRAM consumer GPU optimizations (<16 GB)**, a **built-in open-source 4D Gaussian Splatting (Deformable-GS) studio**, **1-click multi-view background removal**, and **robust hardware telemetry**.

---

## Key Enhancements & New Features

* **Interactive 4-Tab Web UI (`app.py`):**
  * **Tab 1: 4D Video Generation:** Video upload, dynamic VRAM presets, camera orbit settings, and live side-by-side video previews.
  * **Tab 2: 4DGS Reconstruction & Training Studio:** 1-click NeRFStudio continuous dynamic dataset export (`transforms_4d.json` + masks) and integrated Deformable Gaussian Splatting training.
  * **Tab 3: Generated Outputs & Gallery:** Browse past runs, watch 360° videos, inspect metadata, and open folders in File Explorer.
  * **Tab 4: SMPL-X License & Model Manager:** Automated zero-credential body model detection and archive cleanup.
* **1-Click Pinokio App Launcher (`pinokio.js`):**
  * Native 1-click install, launch, update, and reset in [Pinokio](https://pinokio.computer/).
  * Automatic GPU hardware and CUDA detection (RTX 50-series Blackwell, RTX 40-series Ada, RTX 30-series Ampere, Apple Silicon, AMD ROCm).
* **VRAM Optimization Suite (<16 GB / Consumer GPU Ready):**
  * **Proactive Tiled 3D VAE:** Prevents monolithic 28.9 GB VAE allocation spikes, capping peak VRAM to ~8–11 GB.
  * **DiT Block Streaming & TeaCache:** Dynamic CPU<->GPU layer streaming and timestep-skipping for fast inference.
  * **Dynamic Auto-Calibrated Presets:** Low VRAM (<16 GB), Medium VRAM (24 GB), and High VRAM (32 GB+).
* **1-Click Multi-View Character Background Removal:**
  * Extract clean isolated character videos on Green Screen, Pure Black, or Pure White backdrops across all generated camera views via BiRefNet.
* **Open-Source 4D Gaussian Splatting (Deformable-GS):**
  * Native 8-layer sinusoidal deformation field network initialized from Frame 0 visual-hull geometry.
  * Exports standard NeRFStudio / SuperSplat compatible assets with zero proprietary dependencies.
* **Blackwell RTX 5090 & CUDA 12.8 Compatibility:**
  * Fixed non-contiguous 3D memory stride kernel issues during reference camera proposal (RCP) denoising.
* **Smart Zero-Credential SMPL-X Ingestion & Disk Cleaner:**
  * Automatically detects `models_smplx_v1_1.zip` in `~/Downloads` and safely recycles the archive to save disk space.
* **Automated Test Suite & DevOps:**
  * 19 unit tests passing (`run_tests.py`), Dockerfile, Docker Compose, Kubernetes, and GitHub Actions CI/CD workflows.

---

## Quickstart

### Option 1: Pinokio 1-Click Launcher (Easiest)
1. Open [Pinokio](https://pinokio.computer/).
2. Paste `https://github.com/theCosmicCrafter/4DAnyone` into the search bar.
3. Click **Install** -> **Start Web UI**.

### Option 2: Standalone Web UI (Windows / Linux / macOS)
```bash
# Clone the enhanced repository:
git clone https://github.com/theCosmicCrafter/4DAnyone.git
cd 4DAnyone
git submodule update --init third_party/GVHMR

# Setup environment:
conda create -n 4danyone python=3.11 -y
conda activate 4danyone
pip install -r requirements.txt

# Launch interactive Web UI:
python app.py

# Or on Windows, double-click:
start_webui.bat
```

### Verification & Test Suite

Run the full automated test suite (19 unit tests across routing, streaming, quantization, Deformable-GS, and 4D export):

```bash
python run_tests.py
```

For faster inference, optionally install [FlashAttention-3](https://github.com/Dao-AILab/flash-attention/tree/main/hopper) or [SageAttention](https://github.com/thu-ml/SageAttention).

Missing models and examples are downloaded automatically on first use. You can also download them manually:

```bash
python scripts/download_smplx.py
python scripts/download_model.py
python scripts/download_example.py
```

## Inference

4DAnyone supports flexible target-view counts, pitch layers, and yaw coverage. Here are several common camera configurations:

### 6-view full orbit

A compact 360° layout for basic coverage. Start here for an initial test.

```bash
python inference.py \
    --video_path "data/source/pexels/2785536-uhd_2160_3840_25fps.mp4" \
    --views_per_layer 6
```

<p align="left"><img src="docs/assets/inference-6-views.jpg" width="600" alt="Six evenly spaced target cameras on one full orbit"></p>

### 24-view full orbit

A dense 360° layout with broad angular coverage, suitable for 4DGS reconstruction.

```bash
python inference.py \
    --video_path "data/source/pexels/2785536-uhd_2160_3840_25fps.mp4" \
    --views_per_layer 24
```

<p align="left"><img src="docs/assets/inference-24-views.jpg" width="600" alt="Twenty-four evenly spaced target cameras on one full orbit"></p>

### 48-view, three pitch layers

This layout distributes views across three pitch rings for broader coverage, enabling free-viewpoint 4DGS rendering.

```bash
python inference.py \
    --video_path "data/source/pexels/2785536-uhd_2160_3840_25fps.mp4" \
    --views_per_layer 16 --layer_pitches '[-10,15,35]'
```

<p align="left"><img src="docs/assets/inference-48-views-3-layers.jpg" width="600" alt="Forty-eight target cameras arranged over three pitch layers"></p>

### 8-view frontal arc

A focused layout for applications that only require front-side viewpoints.

```bash
python inference.py \
    --video_path "data/source/pexels/2785536-uhd_2160_3840_25fps.mp4" \
    --views_per_layer 8 --start_yaw -90 --yaw_span 180
```

<p align="left"><img src="docs/assets/inference-8-views-front-180.jpg" width="600" alt="Eight target cameras distributed over the frontal 180-degree arc"></p>

### Arguments

Run `python inference.py --help` for the full list. Key camera-layout arguments are:

- `views_per_layer`: number of evenly spaced views per pitch layer; must be divisible by 4 or 6.
- `layer_pitches`: pitch angles in degrees, one per layer; positive values place cameras above the subject. Total views are `views_per_layer × len(layer_pitches)`.
- `start_yaw`: horizontal angle of the first view, in degrees; yaw `0` is the front view.
- `yaw_span`: horizontal range covered by each camera layer, in degrees.

### Output

With the default `--data_dir data`, results follow this layout. See the [output documentation](docs/output.md) for the complete format.

```text
data/
├── gvhmr/results/<clip>/          # reusable motion-recovery result
└── fdanyone/<clip>/
    ├── metadata.json              # run settings, timings, resources
    ├── cameras.json               # the final N-camera rig
    ├── skeletons/00.mp4 ... <N-1>.mp4
    └── videos/
        ├── sparse/{00,04,09,12,14,19}.mp4  # default 24-view RCP proposals
        └── dense/00.mp4 ... <N-1>.mp4       # generated target views
```

### Custom data

Use an input video that:

- is 720p or higher, with 1080p recommended;
- uses a 9:16 portrait aspect ratio;
- shows one person in a full-body or upper-body shot;
- has at least 121 frames;
- contains only mild camera motion.

### Inference Efficiency

See [Inference performance](docs/inference_performance.md) for measured 6-view runtimes and peak GPU memory usage on H20-3E, H200, and RTX A6000 GPUs.

## 3DGS Reconstruction

See the [nerfstudio guide](docs/nerfstudio.md) for details.

## Completed Features & Todos

- [x] **Low-memory inference (<32 GB)**:
  - Resilient auto-tiling VAE encode/decode with OOM exception handling (`fdanyone/model/inference.py`).
  - Daydream FP8 text encoder (`umt5-xxl-enc-fp8_e4m3fn.safetensors`) auto-detection and loading (`fdanyone/model/loader.py`).
  - Variable group sizing (`views_per_group=2, 3, 4, 6`) reducing DiT activation memory (`fdanyone/views.py`).
  - Double-buffered zero-reallocation DiT block streaming with persistent pinned host RAM (`fdanyone/model/streaming.py`).
  - Dynamic FP8 linear weight quantization and ConvRot Hadamard rotation (`fdanyone/model/quantization.py`).
- [x] **Faster inference with TensorRT and sparse attention**:
  - Unified attention kernel dispatcher supporting PyTorch SDPA, SageAttention, FlashAttention, and sparse token routing (`fdanyone/model/attention.py`).
  - TeaCache timestep modulation caching and step-skipping controller (`fdanyone/model/streaming.py`).
- [x] **Support 4DGS reconstruction with an open-source method**:
  - Full-sequence 121-frame multi-view dataset exporter with continuous timestamps (`transforms_4d.json`) and streaming foreground mask generation (`fdanyone/nerfstudio/exporter_4d.py` & `scripts/export_nerfstudio_4d.py`).
  - Canonical Deformable Gaussian Splatting (Deformable-GS) model with 8-layer sinusoidal deformation MLP and visual-hull point cloud initialization (`fdanyone/nerfstudio/deformable_gs.py` & `scripts/train_4dgs.py`).

## Citation

If you find 4DAnyone useful or interesting, please cite our work and consider giving the repository a star:

```bibtex
@article{jin2026fdanyone,
  title={4DAnyone: Create Anyone in 4D from a Casual Monocular Video},
  author={Jin, Yudong and Xie, Tao and Zhang, Qihang and Shen, Zehong and Xu, Zhen and Shen, Yujun and Bao, Hujun and Zhou, Xiaowei and Xu, Yinghao},
  journal={arXiv preprint arXiv:2608.20335},
  year={2026},
  url={https://arxiv.org/abs/2608.20335}
}
```
