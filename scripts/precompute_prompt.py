import os
import sys
import torch
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdanyone.model.loader import ModelLoader

logging.basicConfig(level=logging.INFO)

def precompute():
    print("Initializing ModelLoader to fetch T5...")
    loader = ModelLoader(
        models_dir="models/4danyone",
        device="cuda",
        dtype=torch.bfloat16,
        te_dtype=torch.bfloat16,
        enable_te_fp8=False
    )
    
    print("Loading T5 Text Encoder (this will take a moment and requires ~10GB VRAM)...")
    text_encoder = loader._load_text_encoder(device="cuda")
    
    prompt = "视频中的人在做动作"
    print(f"Encoding fixed prompt: '{prompt}'")
    
    with torch.no_grad():
        context = text_encoder.encode(prompt)
    
    print(f"Embedding shape: {context.shape}")
    
    out_path = os.path.join("models", "4danyone", "prompt_context_fixed.pt")
    torch.save(context.cpu(), out_path)
    
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Successfully saved to {out_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    precompute()
