# PokéLeader — Project Structure, Libraries & Dependencies

---

## Pretrained Models

### 1. Stable Diffusion v1.5
**HuggingFace:** `runwayml/stable-diffusion-v1-5`
**URL:** https://huggingface.co/runwayml/stable-diffusion-v1-5

The core image generation model. Generates the gym leader illustrated portrait from a text prompt. SD 1.5 is chosen over SDXL because:
- IP-Adapter weights are most stable on SD 1.5
- Faster inference on Colab T4 GPU
- Smaller memory footprint
- Better community LoRA support for art styles

---

### 2. IP-Adapter (Face Model)
**HuggingFace:** `h94/IP-Adapter`
**URL:** https://huggingface.co/h94/IP-Adapter
**Weights file used:** `models/ip-adapter-full-face_sd15.bin`

Extracts a face embedding from the user's uploaded photo and injects it into the Stable Diffusion generation process, forcing the output portrait to resemble the user's face while still being fully illustrated.

---

### 3. Pokemon Trainer Style LoRA
**CivitAI:** `Pokemon Trainer Style LoRA`
**URL:** https://civitai.com/models/25997/pokemon-trainer-style-lora
**Trigger word:** `pokemon trainer style`

A community-trained LoRA fine-tuned on official Pokemon game trainer art. Pushes SD 1.5 outputs into the clean, flat-illustrated Pokemon art style. Applied at weight `0.8` during generation.

---

### 4. CLIP Image Encoder (bundled with IP-Adapter)
**HuggingFace:** `openai/clip-vit-large-patch14`
**URL:** https://huggingface.co/openai/clip-vit-large-patch14

Used internally by IP-Adapter to encode the user's face image into an embedding vector. No separate download needed — loaded automatically with IP-Adapter.

---

## Project Folder Structure

```
pokeleader/
│
├── README.md
│
├── data/
│   └── gym_teams.py               # Hardcoded type → Pokemon team mapping
│
├── models/
│   └── (auto-downloaded at runtime via HuggingFace)
│
├── core/
│   ├── face_check.py              # MediaPipe face detection and quality check
│   ├── type_assign.py             # Question answers → gym type logic
│   ├── generate_portrait.py       # SD + IP-Adapter portrait generation
│   └── generate_text.py           # Claude API — opening line + badge name
│
├── assets/
│   ├── card_template.png          # Base card frame (designed by D3)
│   ├── type_badges/               # 8 type badge PNG overlays
│   │   ├── psychic.png
│   │   ├── fighting.png
│   │   ├── normal.png
│   │   ├── fairy.png
│   │   ├── ghost.png
│   │   ├── electric.png
│   │   ├── water.png
│   │   └── dark.png
│   └── fonts/
│       └── pokemon_font.ttf       # Press Start 2P (Google Fonts, free)
│
├── card/
│   ├── fetch_sprites.py           # PokeAPI sprite fetching
│   └── assemble_card.py           # PIL card assembly
│
├── ui/
│   └── app.py                     # Gradio interface
│
├── train/
│   └── config.py                  # Shared constants
│
└── requirements.txt
```

---

## Dependencies

### Python Version
```
Python 3.10+
```

### requirements.txt
```
# Core diffusion
torch==2.1.0
torchvision==0.16.0
diffusers==0.27.2
transformers==4.38.2
accelerate==0.27.2

# IP-Adapter
ip_adapter @ git+https://github.com/tencent-ailab/IP-Adapter.git

# Face detection
mediapipe==0.10.9
opencv-python==4.9.0.80

# Image processing
Pillow==10.2.0

# API and data
anthropic==0.21.3
requests==2.31.0

# UI
gradio==4.19.2

# Utilities
numpy==1.26.4
huggingface_hub==0.21.3
```

### Install Command (Colab)
```python
!pip install torch torchvision diffusers transformers accelerate \
             mediapipe opencv-python Pillow anthropic requests \
             gradio numpy huggingface_hub -q

!pip install git+https://github.com/tencent-ailab/IP-Adapter.git -q
```

---

## Model Download at Runtime

All models are downloaded automatically at first run via HuggingFace Hub:

```python
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
import torch

# Load base SD 1.5
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Download IP-Adapter weights
ip_adapter_path = hf_hub_download(
    repo_id="h94/IP-Adapter",
    filename="models/ip-adapter-full-face_sd15.bin"
)

# Load LoRA from CivitAI (manual download required)
# Download from: https://civitai.com/models/25997
# Save to: /content/pokeleader/models/pokemon_trainer.safetensors
pipe.load_lora_weights("/content/pokeleader/models/pokemon_trainer.safetensors")
```

> **Note:** The Pokemon Trainer LoRA must be manually downloaded from CivitAI and uploaded to Colab or Google Drive. All other models download automatically.

---

## Shared Configuration

**File:** `train/config.py`

```python
# PokéLeader — train/config.py

IMAGE_SIZE      = 512          # SD 1.5 native resolution
NUM_STEPS       = 30           # Inference steps — balance speed vs quality
GUIDANCE_SCALE  = 7.5          # CFG scale
LORA_WEIGHT     = 0.8          # Pokemon art style LoRA strength
IP_ADAPTER_SCALE = 0.7         # Face preservation strength (0=ignore face, 1=exact face)
CLAUDE_MODEL    = "claude-sonnet-4-20250514"
MAX_TOKENS      = 300
```

---

## External APIs

### PokeAPI
- **URL:** https://pokeapi.co
- **Auth:** None required
- **Usage:** Fetch Pokemon sprites by name at runtime
- **Rate limit:** 100 requests/minute (well within project needs)

### Anthropic Claude API
- **Model:** `claude-sonnet-4-20250514`
- **Auth:** API key via environment variable `ANTHROPIC_API_KEY`
- **Usage:** Generate opening trash talk line and badge name

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "your_key_here"  # Set in Colab secrets
```

---

## Google Drive Folder Structure

All outputs and the manually downloaded LoRA are stored in shared Google Drive:

```
MyDrive/pokeleader/
├── models/
│   └── pokemon_trainer.safetensors    # Manual CivitAI download
├── outputs/
│   └── (generated cards saved here)
└── assets/
    └── (card template and badges)
```

---

*PokéLeader — Project Structure v1.0*
