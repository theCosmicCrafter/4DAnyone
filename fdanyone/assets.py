"""Locate the model files used by 4DAnyone.

Every published file is anchored by one immutable Hugging Face revision and
downloaded on demand. ``fdanyone.download`` fetches missing files;
the resolvers here only locate them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fdanyone.errors import AssetError

HF_REPO_ID = "AntResearch/4DAnyone"
HF_REVISION = "7850985888b56aabf09e69480b73248f1a76bcbe"

BIREFNET_REPO_ID = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
BIREFNET_DIR = "birefnet"
BIREFNET_FILES = (
    "BiRefNet_config.py",
    "birefnet.py",
    "config.json",
    "model.safetensors",
)

CHECKPOINT = "4danyone/model.safetensors"
MHR70_REGRESSOR = "4danyone/smplx_to_goliath70.pt"
WAN_VAE = "4danyone/Wan2.2_VAE.pth"
TEXT_ENCODER = "4danyone/models_t5_umt5-xxl-enc-bf16.pth"
TEXT_ENCODER_FP8 = "4danyone/umt5-xxl-enc-fp8_e4m3fn.safetensors"
TOKENIZER_DIR = "4danyone/umt5-xxl"
TOKENIZER_FILES = tuple(
    f"{TOKENIZER_DIR}/{name}"
    for name in ("special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json")
)

GVHMR_CHECKPOINT = "gvhmr/gvhmr_siga24_release.ckpt"
HMR2_CHECKPOINT = "gvhmr/epoch=10-step=25000.ckpt"
VITPOSE_CHECKPOINT = "gvhmr/vitpose-h-multi-coco.pth"
YOLO_CHECKPOINT = "gvhmr/yolov8x.pt"
PERCEPTUAL_VGG19 = "perceptual/imagenet-vgg-verydeep-19-conv.safetensors"

SMPLX_MODEL = "body_models/smplx/SMPLX_NEUTRAL.npz"

MODEL_FILES = (
    CHECKPOINT,
    MHR70_REGRESSOR,
    WAN_VAE,
    # TEXT_ENCODER is bypassed via precomputed prompt_context_fixed.pt
    # TEXT_ENCODER,
    *TOKENIZER_FILES,
    GVHMR_CHECKPOINT,
    HMR2_CHECKPOINT,
    VITPOSE_CHECKPOINT,
    PERCEPTUAL_VGG19,
)

EXAMPLE_FILES = (
    "data/source/pexels/10331522-uhd_2160_4096_25fps.mp4",
    "data/source/pexels/2785536-uhd_2160_3840_25fps.mp4",
    "data/source/pexels/5435720-uhd_2160_4096_25fps.mp4",
    "data/source/pexels/5885633-hd_1080_1920_25fps.mp4",
    "data/source/pexels/5999210-uhd_2160_4096_25fps.mp4",
    "data/source/pexels/6980035-uhd_2160_4096_30fps.mp4",
    "data/source/pexels/7080903-hd_1080_1920_30fps.mp4",
    "data/source/pexels/7480858-uhd_2160_3840_25fps.mp4",
)

# Upstream GVHMR resolves its model files relative to its own checkout, so the
# install commands link each downloaded file to the location GVHMR expects.
GVHMR_LINKS = (
    (GVHMR_CHECKPOINT, "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"),
    (HMR2_CHECKPOINT, "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"),
    (VITPOSE_CHECKPOINT, "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"),
    (SMPLX_MODEL, "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"),
)


@dataclass(frozen=True)
class BaseAssets:
    vae: Path
    text_encoder: Path
    tokenizer: Path


def _require_file(path: Path, label: str, command: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise AssetError(f"{label} does not exist: {resolved}. Run `python {command}` to install it.")
    return resolved


def resolve_checkpoint(path: str | Path | None = None, model_dir: str | Path = "models") -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise AssetError(f"Checkpoint override does not exist: {resolved}")
        return resolved
        
    int8_path = Path(model_dir).expanduser().resolve() / "4danyone/model_int8_convrot.safetensors"
    if int8_path.is_file():
        return int8_path.resolve()
        
    return _require_file(Path(model_dir) / CHECKPOINT, "Checkpoint", "scripts/download_model.py")


def resolve_regressor(path: str | Path | None = None, model_dir: str | Path = "models") -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise AssetError(f"MHR70 regressor override does not exist: {resolved}")
        return resolved
    return _require_file(Path(model_dir) / MHR70_REGRESSOR, "MHR70 regressor", "scripts/download_model.py")


def resolve_foreground_model(model_dir: str | Path = "models") -> Path:
    root = Path(model_dir).expanduser() / BIREFNET_DIR
    for relative in BIREFNET_FILES:
        _require_file(root / relative, "BiRefNet file", "scripts/download_model.py")
    return root.resolve()


def resolve_perceptual_vgg19(model_dir: str | Path = "models") -> Path:
    """Resolve the converted VGG-19 weights used by perceptual reconstruction."""

    return _require_file(
        Path(model_dir) / PERCEPTUAL_VGG19,
        "Perceptual VGG-19 weights",
        "scripts/download_model.py",
    )


def resolve_base_assets(model_dir: str | Path = "models", prefer_fp8: bool = True) -> BaseAssets:
    """Resolve the local VAE, T5 encoder, and tokenizer."""

    root = Path(model_dir).expanduser()
    prompt_context = root / "4danyone/prompt_context_fixed.pt"
    
    if not prompt_context.is_file():
        for relative in TOKENIZER_FILES:
            _require_file(root / relative, "Tokenizer file", "scripts/download_model.py")

    text_encoder_fp8 = root / TEXT_ENCODER_FP8
    if prompt_context.is_file():
        text_encoder = prompt_context.resolve()
    elif prefer_fp8 and text_encoder_fp8.is_file():
        text_encoder = text_encoder_fp8.resolve()
    else:
        text_encoder = _require_file(root / TEXT_ENCODER, "Text encoder", "scripts/download_model.py")

    return BaseAssets(
        vae=_require_file(root / WAN_VAE, "VAE", "scripts/download_model.py"),
        text_encoder=text_encoder,
        tokenizer=(root / TOKENIZER_DIR).expanduser().resolve() if (root / TOKENIZER_DIR).exists() else root,
    )
