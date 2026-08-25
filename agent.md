# 4DAnyone — Master Agent Implementation & Architecture Blueprint

> This document provides the complete architecture blueprint, technical specifications, and prioritized implementation plans for optimizing **4DAnyone**. It incorporates state-of-the-art memory engineering patterns from **ComfyUI**, **Wan2GP (`mmgp`)**, **Daydream Scope / Livepeer**, and **ConvRot (W4A4/W4A8)**.

---

## 1. Project Context & Current Architecture

**4DAnyone** generates synchronized multi-view videos from casual monocular video to power downstream 4D Gaussian Splatting (4DGS) reconstruction.

### Current Pipeline Overview (`fdanyone/`)
```
inference.py (Fire CLI Entrypoint)
└── fdanyone/pipeline.py (Orchestrator)
    ├── fdanyone/motion/worker.py       [Subprocess] GVHMR SMPL-X motion recovery
    ├── fdanyone/skeleton/worker.py     [Subprocess] BiRefNet masks + MHR70 depth-buffered rasterization
    └── fdanyone/model/inference.py     [In-Process] Multi-view Wan2.2 DiT Generation
        ├── fdanyone/model/loader.py    Meta-tensor lazy loading (DiT 5B + 3D VAE + T5-XXL)
        ├── fdanyone/model/routing.py   Target-Context Routing (TCR) view permutations
        └── fdanyone/vendor/diffsynth/  Vendored DiT, 38-ch Causal VAE, FlowMatch Scheduler
```

### Numerical Contracts & Baseline Footprints
* **Frames**: Exactly 121 frames at $1280 \times 704$ resolution.
* **Latent Tensor**: $[B, 16, 31, 160, 88]$ (downsampled $4\times$ temporal, $8\times$ spatial).
* **DiT Dimension**: 5B parameters (dim=3072, 30 layers, 24 heads, in_dim=48, out_dim=48).
* **Baseline VRAM Usage**: **32 GB – 48 GB peak** (requires A100/H100 or dual GPUs).

---

## 2. Deep VRAM & Speed Research Synthesis

### A. ComfyUI Memory Management Patterns
* **Soft VRAM Budgeting**:
  $$\text{VRAM}_{\text{weights\_max}} = \text{VRAM}_{\text{total}} \times 0.88 - \text{Memory}_{\text{min\_inference}}$$
  where $\text{Memory}_{\text{min\_inference}} \approx 800\text{ MB} + \text{Reserved (600 MB Windows / 400 MB Linux)}$.
* **Auto-Tiling VAE Fallback**: Catches `torch.cuda.OutOfMemoryError` during standard VAE decoding, triggers `torch.cuda.empty_cache()`, and retries with 3D spatio-temporal sliding tile decoding.
* **Sub-Module Weight Casting (`comfy_cast_weights`)**: Sorts modules by memory footprint and keeps high-priority blocks resident on GPU while dynamically streaming the rest over PCIe during the forward pass.

### B. Wan2GP (`mmgp`) Engine Patterns
* **Block-by-Block DiT Streaming**: Keeps only 1–4 transformer blocks in GPU VRAM at any moment. For 4DAnyone's 30-block DiT, weight footprint drops from **10.5 GB to ~1.4 GB**.
* **Double-Buffered CUDA Stream Prefetching**: Uses a `compute_stream` and `copy_stream`. While Block $N$ evaluates on GPU, Block $N+1$ prefetches asynchronously from pinned host RAM via direct memory access (DMA).
* **Pinned System RAM (`param.data.pin_memory()`)**: Delivers 90%+ PCIe throughput, limiting the streaming penalty to <15–20% latency overhead.
* **TeaCache / First Block Cache**: Checks delta in modulation layers or Block 0. If latent variance is below threshold across diffusion timesteps, intermediate block evaluations are skipped.

### C. Daydream Scope / Livepeer Innovations
* **Daydream FP8 UMT5-XXL Checkpoint (`daydreamlive/Wan2.1-T2V-14B`)**:
  * Checkpoint: `umt5-xxl-enc-fp8_e4m3fn.safetensors`
  * Reduces text encoder memory from **9.5 GB (BF16) down to ~4.8 GB (FP8)**.
* **VACE Parallel Conditioning Adaptation ([arXiv:2602.14381](https://arxiv.org/abs/2602.14381))**:
  * Solved structural video-to-video control by routing conditioning (pose, depth, masks) through a parallel conditioning pathway rather than concatenating into latent channels. Preserves KV-cache persistence with zero retraining needed.
* **TensorRT 10.x Static Shape Engines**:
  * Scope compiles DiT chunk passes to TensorRT with fixed batch shapes, achieving **17–22 FPS** streaming generation.

### D. ConvRot (INT4 / W4A4 / SVDQuant)
* **Hadamard Orthogonal Rotation**: Eliminates activation channel outliers in Diffusion Transformers before quantization.
* **Impact**: Compresses DiT weights to **~2.7 GB (INT4)** while maintaining visual parity with BF16.
* **4DAnyone Adaptation**: Requires running the ConvRot offline calibration script directly on 4DAnyone's custom multi-view `model.safetensors`.

---

## 3. Phased Implementation Roadmap

```mermaid
graph TD
    subgraph "Phase 1: Immediate Drop-in (14-18 GB VRAM)"
        P1A["Auto-Tiling VAE with OOM Fallback"] --> P1B["Daydream FP8 UMT5-XXL Text Encoder"]
        P1B --> P1C["Group Size Reductions (2, 3)"]
    end

    subgraph "Phase 2: Consumer Tier (10-12 GB VRAM)"
        P2A["mmgp Block-by-Block DiT Streaming"] --> P2B["Pinned RAM Double-Buffering"]
        P2B --> P2C["FP8 DiT Quantization via torchao"]
        P2C --> P2D["TeaCache Step Skipping"]
    end

    subgraph "Phase 3: Ultra Performance (6-8 GB VRAM / 3x Speed)"
        P3A["ConvRot W4A4 Calibration on model.safetensors"] --> P3B["TensorRT / torch.compile Static Graph"]
        P3B --> P3C["SageAttention / Sol-Attn Kernel Selection"]
    end

    subgraph "Phase 4: Open Source 4DGS Reconstruction"
        P4A["Full 121-Frame Multi-View Exporter"] --> P4B["Deformable-GS Dynamic 4D Pipeline"]
        P4B --> P4C["VGG19 Perceptual Temporal Loss Integration"]
    end

    Phase 1 --> Phase 2
    Phase 2 --> Phase 3
    Phase 3 --> Phase 4
```

---

## 4. Detailed Implementation Instructions

### Phase 1: Immediate Drop-in (Target: 14–18 GB VRAM)

#### 1.1 Auto-Tiling VAE Fallback in `fdanyone/model/inference.py`
Replace static `_tiler_kwargs()` calls with an adaptive try-except block:
```python
# fdanyone/model/inference.py
def _decode_video_adaptive(pipe, latents, device: str):
    import torch
    try:
        # Attempt direct decode for maximum speed
        return pipe.decode_video(latents.to(dtype=pipe.torch_dtype, device=device), tiled=False)[0]
    except torch.cuda.OutOfMemoryError:
        _empty_cuda_cache()
        LOGGER.warning("OOM during full VAE decode. Falling back to tiled 3D decoding.")
        return pipe.decode_video(
            latents.to(dtype=pipe.torch_dtype, device=device),
            tiled=True,
            tile_size=INFERENCE.vae_tile_size,
            tile_stride=INFERENCE.vae_tile_stride,
        )[0]
```

#### 1.2 Daydream FP8 UMT5-XXL Text Encoder in `fdanyone/model/loader.py`
Add support for loading FP8 text encoder weights:
```python
# fdanyone/assets.py
T5_FP8_FILENAME = "umt5-xxl-enc-fp8_e4m3fn.safetensors"
DAYDREAM_REPO_ID = "daydreamlive/Wan2.1-T2V-14B"

# fdanyone/model/loader.py
def load_text_encoder(assets: BaseAssets, device: str, use_fp8: bool = True):
    if use_fp8:
        # Load Daydream FP8 UMT5 weights directly (cuts VRAM from 9.5GB to 4.8GB)
        ...
```

#### 1.3 Variable Group Sizing in `fdanyone/views.py` & `fdanyone/model/routing.py`
* Allow `views_per_group` to accept `2`, `3`, `4`, or `6`.
* Adjust TCR routing offsets: `offset = (step_index * (group_size - 1)) % views_per_layer`.

---

### Phase 2: Consumer Tier (Target: 10–12 GB VRAM)

#### 2.1 `mmgp` Block-by-Block DiT Streaming
Implement a double-buffered block streamer in `fdanyone/model/inference.py`:
```python
class DiTBlockStreamer:
    def __init__(self, dit_blocks, device: str):
        self.blocks = dit_blocks
        self.device = device
        self.copy_stream = torch.cuda.Stream()
        # Pin CPU parameters
        for block in self.blocks:
            for p in block.parameters():
                p.data = p.data.pin_memory()

    def stream_forward(self, x, x_src, timestep, skeletons, context, extra_inputs):
        compute_stream = torch.cuda.current_stream()
        # Preload block 0
        self.blocks[0].to(self.device, non_blocking=True)
        
        for idx in range(len(self.blocks)):
            compute_stream.wait_stream(self.copy_stream)
            # Async prefetch block idx + 1
            if idx + 1 < len(self.blocks):
                with torch.cuda.stream(self.copy_stream):
                    self.blocks[idx + 1].to(self.device, non_blocking=True)
            
            # Execute block idx
            x = self.blocks[idx](x, x_src, timestep, skeletons, **context, **extra_inputs)
            
            # Evict block idx
            self.blocks[idx].to("cpu", non_blocking=True)
        return x
```

#### 2.2 TeaCache Step Skipping in Denoising Loop
Add delta monitoring between timesteps in `_denoise_targets`:
```python
# Skip redundant intermediate block passes when modulation change is < epsilon
if step_index > 0 and modulation_delta < TEACACHE_THRESHOLD:
    latents = apply_cached_residual(latents, cached_prediction, timestep)
```

---

### Phase 3: Ultra Performance & 8GB Tier (Target: 6–8 GB VRAM, 3x Speed)

#### 3.1 Offline ConvRot (W4A4) Calibration on `model.safetensors`
1. Create `scripts/quantize_convrot.py`.
2. Compute Hadamard rotation matrices for 4DAnyone's 5B DiT linear layers.
3. Save rotated & packed 4-bit weights as `model_convrot_w4a4.safetensors`.
4. Integrate `ComfyUI-INT4-Fast` / `torchao` INT4 GEMM kernels in `wan_video_dit.py`.

#### 3.2 TensorRT Static Engine Compilation
* 4DAnyone's DiT operates on strictly static shapes ($31 \times 80 \times 44$ latents).
* Compile individual blocks or the full backbone using `torch.compile(backend="tensorrt")` with static CUDA execution buffers.

---

### Phase 4: Open Source 4DGS Reconstruction

#### 4.1 Full-Sequence Multi-View Exporter (`fdanyone/nerfstudio/exporter_4d.py`)
* Extend exporter to iterate over all $t \in [0, 120]$.
* Output structured datasets:
  ```
  data/fdanyone/<clip>/transforms_4d.json
  data/fdanyone/<clip>/images/cam_XX_frame_YYY.png
  data/fdanyone/<clip>/masks/cam_XX_frame_YYY.png
  ```
* Set camera timestamp metadata: `"time": frame_idx / 120.0`.

#### 4.2 Deformable-GS Model Integration
* **Canonical 3D Gaussians**: Initialized from `sparse_pcd.ply` (visual hull).
* **Deformation MLP**: $\Delta \mu, \Delta r, \Delta s = \text{MLP}(x, y, z, t)$.
* **Temporal Perceptual Loss**: Apply `VGG19PerceptualLoss` across rendered RGB frames and ground-truth generated multi-view videos.

---

## 5. Verification & Testing Matrix

| Component | Test Command / Procedure | Pass Criteria |
|---|---|---|
| **CLI & Help** | `python inference.py --help` | Exits 0 with all flags documented |
| **FP8 Text Encoder** | `pytest tests/test_loader.py` | VRAM peak < 5.0 GB during prompt encoding |
| **Tiled VAE Fallback** | Run inference with simulated low VRAM | Fallback triggered cleanly without crash |
| **Block Streamer** | Run 6-view generation on 16GB GPU | Peak VRAM < 12 GB, output MP4 valid |
| **Pinokio Launcher** | Run `pterm status 4DAnyone` | Launcher online, form functional |
| **4DGS Export** | `python scripts/export_nerfstudio_4d.py --clip test` | `transforms_4d.json` created with 121 timestamps |
