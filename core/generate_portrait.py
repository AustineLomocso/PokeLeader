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
        # Load base pipeline
        _pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")

        # Load Pokemon Trainer LoRA if it exists
        if os.path.exists(LORA_PATH):
            _pipe.load_lora_weights(LORA_PATH)
            _pipe.fuse_lora(lora_scale=LORA_WEIGHT)

        # Load IP-Adapter face weights
        ip_adapter_weights = hf_hub_download(
            repo_id="h94/IP-Adapter",
            filename="models/ip-adapter-full-face_sd15.bin"
        )

        _ip_model = IPAdapterFull(
            sd_pipe=_pipe,
            image_encoder_path="openai/clip-vit-large-patch14",
            ip_ckpt=ip_adapter_weights,
            device="cuda"
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

    prompt = (
        f"pokemon trainer style, gym leader portrait, "
        f"{type_details}, "
        f"official pokemon game art style, clean illustration, "
        f"detailed character portrait, vibrant colors, no text, no watermark"
    )

    face_pil = Image.fromarray(face_image_array.astype(np.uint8))

    images = ip_model.generate(
        pil_image=face_pil,
        num_samples=1,
        num_inference_steps=NUM_STEPS,
        seed=42,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        scale=IP_ADAPTER_SCALE,
        guidance_scale=GUIDANCE_SCALE,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
    )

    return images[0]
