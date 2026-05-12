# PokéLeader — core/generate_portrait.py

import sys
import torch
from diffusers import StableDiffusionPipeline
from ip_adapter import IPAdapterFull  
from huggingface_hub import hf_hub_download
from train.config import IMAGE_SIZE, NUM_STEPS, GUIDANCE_SCALE, LORA_WEIGHT, IP_ADAPTER_SCALE
import os
from PIL import Image
import numpy as np

# Adjust based on environment
DRIVE_BASE = os.getcwd() # Using current directory for local dev
LORA_PATH  = os.path.join(DRIVE_BASE, 'models', 'pokemon_trainer.safetensors')

# Lazy loading to avoid errors if models aren't present yet during file generation
_pipe = None
_ip_model = None

def get_pipeline():
    global _pipe, _ip_model
    if _pipe is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        # Load base pipeline
        _pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            torch_dtype=dtype,
            variant="fp16",
            safety_checker=None,
            low_cpu_mem_usage=True,
        ).to(device)
        _pipe.upcast_vae()
        # Optimizations for low VRAM/RAM
        _pipe.enable_attention_slicing()

        # Load Pokemon Trainer LoRA if it exists
        if os.path.exists(LORA_PATH):
            try:
                _pipe.load_lora_weights(LORA_PATH)
                _pipe.fuse_lora(lora_scale=LORA_WEIGHT)
                print("✓ Pokemon Trainer LoRA loaded and fused.")
            except Exception as e:
                print(f"⚠ Could not load LoRA: {e}. Continuing without it.")
        ip_adapter_weights = hf_hub_download(
            repo_id="h94/IP-Adapter",
            filename="models/ip-adapter-full-face_sd15.bin"  
        )

        _ip_model = IPAdapterFull(  
            sd_pipe=_pipe,
            image_encoder_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",  
            ip_ckpt=ip_adapter_weights,
            device=device
        )
    return _ip_model

TYPE_PROMPTS = {
    'psychic':  'dramatic purple pink lighting, mystical aura, calm intense eyes',
    'fighting': 'warm red orange lighting, strong confident pose, determined expression',
    'normal':   'soft neutral lighting, approachable expression, clean background',
    'fairy':    'pastel pink lighting, whimsical magical aura, playful expression',
    'ghost':    'dark dramatic lighting, mysterious shadowy atmosphere, ethereal glow',
    'electric': 'bright yellow white lighting, electric sparks, energetic expression',
    'water':    'cool blue teal lighting, calm composed expression, flowing atmosphere',
    'dark':     'dark grey blue lighting, stoic expression, minimal dramatic shadows',
}

NEGATIVE_PROMPT = (
    "deformed, ugly, blurry, low quality, extra limbs, "
    "disfigured, bad anatomy, watermark, text, logo"
)


def generate_portrait(face_image_array, gym_type):
    """
    Generates a gym leader portrait with the user's face.
    """
    ip_model = get_pipeline()
    
    type_details = TYPE_PROMPTS.get(gym_type, '')

    # FIX 4: Add explicit subject tags to prevent anatomy hallucinations
    prompt = (
        f"1person, solo, upper body portrait of a pokemon trainer, gym leader, "
        f"{type_details}, "
        f"official pokemon game art style, clean illustration, "
        f"detailed character portrait, vibrant colors, no text, no watermark"
    )

    # Ensure it's in RGB (just an extra safety net)
    face_pil = Image.fromarray(face_image_array.astype(np.uint8)).convert("RGB")
    face_pil.save("debug_crop_check.png")
    images = ip_model.generate(
        pil_image=face_pil,
        num_samples=1,
        num_inference_steps=NUM_STEPS,
        seed=42, # Consider making this random later so users don't get the same pose every time!
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        scale=IP_ADAPTER_SCALE,
        guidance_scale=GUIDANCE_SCALE,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
    )

    return images[0]