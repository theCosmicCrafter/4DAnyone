import os
import sys
import torch
import logging
from pathlib import Path

# Fix paths for Pinokio environment
app_dir = Path("C:/pinokio/api/4DAnyone/app")
sys.path.append(str(app_dir))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fdanyone.model.loader import _load_text_encoder
from fdanyone import assets
from fdanyone.vendor.diffsynth.prompters.wan_prompter import WanPrompter

logging.basicConfig(level=logging.INFO)

def precompute():
    print("Loading T5 Text Encoder (this will take a moment and requires ~10GB VRAM)...")
    text_encoder_path = app_dir / "models" / assets.TEXT_ENCODER
    text_encoder = _load_text_encoder(text_encoder_path, torch.bfloat16)
    
    # Must move to CUDA before embedding!
    text_encoder.to("cuda")
    
    tokenizer_dir = app_dir / "models" / assets.TOKENIZER_DIR
    prompter = WanPrompter(tokenizer_path=str(tokenizer_dir))
    prompter.fetch_models(text_encoder)
    
    prompt = "????,??,??,??????,??,??,??,??,??,??,????,????,???,JPEG????,???,???,?????,???????,???????,???,???,???????,????,???????,?????,???,?????,???"
    print("Encoding fixed negative prompt...")
    
    with torch.no_grad():
        context = prompter.encode_prompt(prompt, positive=False)
        
    print(f"Embedding shape: {context.shape}")
    
    out_path = app_dir / "models" / "4danyone" / "prompt_context_fixed.pt"
    torch.save(context.cpu(), out_path)
    
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Successfully saved to {out_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    precompute()

