import os
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline
import torch

def download_all():
    print("--- Starting Model Downloads ---")
    
    # 1. Download Stable Diffusion v1.5
    print("\n[1/2] Downloading Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5 - fp16)...")
    try:
        StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
            # We don't need the safety checker for the download phase
            safety_checker=None 
        )
        print("✓ Stable Diffusion v1.5 downloaded.")
    except Exception as e:
        print(f"✗ Failed to download SD 1.5: {e}")

    # 2. Download IP-Adapter Face Weights
    print("\n[2/2] Downloading IP-Adapter Face Weights (h94/IP-Adapter)...")
    try:
        hf_hub_download(
            repo_id="h94/IP-Adapter",
            filename="models/ip-adapter-full-face_sd15.bin"
        )
        print("✓ IP-Adapter weights downloaded.")
    except Exception as e:
        print(f"✗ Failed to download IP-Adapter: {e}")

    print("\n--- HuggingFace Downloads Complete ---")
    print("Note: The 'Pokemon Trainer Style LoRA' must still be manually downloaded from CivitAI.")

if __name__ == "__main__":
    download_all()
