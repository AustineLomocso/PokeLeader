# PokéLeader — 1-Day Task Flow (3 Developers)

---

## Developer Assignments

| Developer | Role | Branch |
|---|---|---|
| D1 | Face Pipeline + Type Logic | `dev/d1-face-type` |
| D2 | Portrait Generation | `dev/d2-portrait` |
| D3 | Card Assembly + UI | `dev/d3-card-ui` |

---

## Timeline Overview

```
Hour 1      All — Environment setup, branching, Drive connection
            │
Hour 2–3    D1 — Face detection + type assignment logic
            D2 — SD + IP-Adapter setup + prompt testing
            D3 — Card template design + sprite fetching
            │
Hour 4      D1 — Finalize and push face encoder output
            D2 — Finalize portrait generation function
            D3 — Claude API text generation
            │
            ▼ INTEGRATION POINT — All three outputs merge
            │
Hour 5      D3 — Wire everything into Gradio UI
            D1 — Support integration, fix bugs
            D2 — Support integration, tune portrait quality
            │
            ▼ END OF DAY — Full demo running
```

---

## Hour 1 — All Developers (Environment Setup)

> Everyone completes this before starting their assigned tasks.

### Step 1 — Clone Repository
```python
!git clone https://github.com/YOUR_ORG/pokeleader.git
%cd pokeleader
```

### Step 2 — Configure Git Identity
```python
%%bash
git config user.email "your@email.com"
git config user.name "Your Name"
git remote set-url origin https://YOUR_USERNAME:YOUR_TOKEN@github.com/YOUR_ORG/pokeleader.git
```

### Step 3 — Install Dependencies
```python
!pip install torch torchvision diffusers transformers accelerate \
             mediapipe opencv-python Pillow anthropic requests \
             gradio numpy huggingface_hub -q
!pip install git+https://github.com/tencent-ailab/IP-Adapter.git -q
```

### Step 4 — Create Your Branch
```python
# D1
%%bash
git checkout -b dev/d1-face-type
git push -u origin dev/d1-face-type

# D2
%%bash
git checkout -b dev/d2-portrait
git push -u origin dev/d2-portrait

# D3
%%bash
git checkout -b dev/d3-card-ui
git push -u origin dev/d3-card-ui
```

### Step 5 — Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 6 — Create Shared Folder Structure
```python
import os
base = '/content/drive/MyDrive/pokeleader'
folders = ['models', 'outputs', 'assets/type_badges', 'assets/fonts']
for f in folders:
    os.makedirs(os.path.join(base, f), exist_ok=True)
print('Drive folders ready')
```

> ✅ Hour 1 complete. Everyone starts their assigned tasks simultaneously.

---

## Hours 2–4 — Developer 1: Face Pipeline + Type Logic

> **File outputs:** `core/face_check.py` · `core/type_assign.py` · `data/gym_teams.py`

### Task D1.1 — Face Detection and Quality Check
**File:** `core/face_check.py`

```python
# PokéLeader — core/face_check.py

import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection


def check_and_crop_face(image_array):
    """
    Accepts a numpy array (RGB, from Gradio).
    Returns a cropped face numpy array (RGB, 256x256)
    or None if no face detected or quality too low.
    """
    h, w = image_array.shape[:2]

    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as detector:
        results = detector.process(image_array)

    if not results.detections:
        return None, "No face detected. Please upload a clear front-facing photo."

    # Use the first (most confident) detection
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box

    # Convert relative to absolute coordinates
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    # Add padding around the face
    pad = int(max(bw, bh) * 0.3)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    face_crop = image_array[y1:y2, x1:x2]

    # Quality check — face too small
    if face_crop.shape[0] < 100 or face_crop.shape[1] < 100:
        return None, "Face too small. Please upload a closer photo."

    # Resize to standard size for IP-Adapter
    face_resized = cv2.resize(face_crop, (256, 256))
    return face_resized, "ok"
```

---

### Task D1.2 — Gym Type Assignment Logic
**File:** `core/type_assign.py`

```python
# PokéLeader — core/type_assign.py

# Answer index mapping
# Question 1 — Villain origin: 0=wifi, 1=food, 2=advice, 3=lost
# Question 2 — Habitat: 0=coffee, 1=gym5am, 2=bed, 3=uninvited
# Question 3 — Group project: 0=did_everything, 1=one_slide, 2=was_vibe, 3=disappeared

TYPE_MAP = {
    (0, 0, 0): 'psychic',    # wifi + coffee + did everything
    (0, 0, 2): 'electric',   # wifi + coffee + was the vibe
    (0, 1, 0): 'fighting',   # wifi + gym5am + did everything
    (0, 1, 3): 'dark',       # wifi + gym5am + disappeared
    (0, 2, 1): 'normal',     # wifi + bed + one slide
    (0, 2, 2): 'electric',   # wifi + bed + was the vibe
    (1, 0, 0): 'psychic',    # food + coffee + did everything
    (1, 1, 0): 'fighting',   # food + gym5am + did everything
    (1, 2, 1): 'normal',     # food + bed + one slide
    (1, 3, 2): 'fairy',      # food + uninvited + was the vibe
    (2, 0, 0): 'psychic',    # advice + coffee + did everything
    (2, 3, 2): 'fairy',      # advice + uninvited + was the vibe
    (3, 0, 0): 'ghost',      # lost + coffee + did everything
    (3, 2, 2): 'electric',   # lost + bed + was the vibe
    (3, 3, 3): 'dark',       # lost + uninvited + disappeared
}

# Fallback mapping when exact combo not found
FALLBACK_BY_Q1 = {
    0: 'electric',
    1: 'water',
    2: 'fairy',
    3: 'ghost',
}

TYPE_COLORS = {
    'psychic':  '#FF6FA0',
    'fighting': '#C03028',
    'normal':   '#A8A878',
    'fairy':    '#EE99AC',
    'ghost':    '#705898',
    'electric': '#F8D030',
    'water':    '#6890F0',
    'dark':     '#705848',
}


def assign_type(q1_idx, q2_idx, q3_idx):
    """
    Takes three answer indices (0-3 each).
    Returns gym type string and its hex color.
    """
    key = (q1_idx, q2_idx, q3_idx)
    gym_type = TYPE_MAP.get(key, FALLBACK_BY_Q1.get(q1_idx, 'normal'))
    color = TYPE_COLORS[gym_type]
    return gym_type, color
```

---

### Task D1.3 — Gym Team Mapping
**File:** `data/gym_teams.py`

```python
# PokéLeader — data/gym_teams.py

GYM_TEAMS = {
    'psychic':  ['alakazam', 'gardevoir', 'espeon'],
    'fighting': ['machamp', 'lucario', 'hawlucha'],
    'normal':   ['snorlax', 'staraptor', 'blissey'],
    'fairy':    ['togekiss', 'sylveon', 'clefable'],
    'ghost':    ['gengar', 'chandelure', 'mimikyu'],
    'electric': ['jolteon', 'magnezone', 'rotom'],
    'water':    ['vaporeon', 'starmie', 'gyarados'],
    'dark':     ['umbreon', 'hydreigon', 'weavile'],
}
```

### Push D1 Work
```python
%%bash
git add core/face_check.py core/type_assign.py data/gym_teams.py
git commit -m "feat(d1): face detection, type assignment logic, gym team mapping"
git push origin dev/d1-face-type
```

> ✅ D1 complete. Notify D3 that face crop output and type logic are ready.

---

## Hours 2–4 — Developer 2: Portrait Generation

> **File output:** `core/generate_portrait.py`

### Task D2.1 — Load SD + IP-Adapter Pipeline
```python
# PokéLeader — core/generate_portrait.py

import sys
sys.path.append('/content/pokeleader')

import torch
from diffusers import StableDiffusionPipeline
from ip_adapter import IPAdapterFull
from huggingface_hub import hf_hub_download
from train.config import IMAGE_SIZE, NUM_STEPS, GUIDANCE_SCALE, LORA_WEIGHT, IP_ADAPTER_SCALE

DRIVE_BASE = '/content/drive/MyDrive/pokeleader'
LORA_PATH  = f'{DRIVE_BASE}/models/pokemon_trainer.safetensors'

# Load base pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

# Load Pokemon Trainer LoRA
pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=LORA_WEIGHT)

# Load IP-Adapter face weights
ip_adapter_weights = hf_hub_download(
    repo_id="h94/IP-Adapter",
    filename="models/ip-adapter-full-face_sd15.bin"
)

ip_model = IPAdapterFull(
    sd_pipe=pipe,
    image_encoder_path="openai/clip-vit-large-patch14",
    ip_ckpt=ip_adapter_weights,
    device="cuda"
)
```

### Task D2.2 — Write Portrait Generation Function

```python
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

    Args:
        face_image_array : 256x256 numpy array (RGB) from D1 face check
        gym_type         : string gym type from D1 type assignment

    Returns:
        PIL Image of the generated portrait (512x512)
    """
    type_details = TYPE_PROMPTS.get(gym_type, '')

    prompt = (
        f"pokemon trainer style, gym leader portrait, "
        f"{type_details}, "
        f"official pokemon game art style, clean illustration, "
        f"detailed character portrait, vibrant colors, no text, no watermark"
    )

    from PIL import Image
    import numpy as np
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
```

### Task D2.3 — Test Portrait Quality

Test with at least 3 different gym types before pushing. Adjust `IP_ADAPTER_SCALE` in config if:
- Face not recognizable → increase scale toward 0.9
- Output looks too photorealistic → decrease LoRA weight toward 0.6

### Push D2 Work
```python
%%bash
git add core/generate_portrait.py train/config.py
git commit -m "feat(d2): SD + IP-Adapter portrait generation pipeline"
git push origin dev/d2-portrait
```

> ✅ D2 complete. Notify D3 that portrait generation function is ready.

---

## Hours 2–4 — Developer 3: Card Assembly + Claude API

> **File outputs:** `core/generate_text.py` · `card/fetch_sprites.py` · `card/assemble_card.py`

### Task D3.1 — Claude API Text Generation
**File:** `core/generate_text.py`

```python
# PokéLeader — core/generate_text.py

import anthropic
import os
from train.config import CLAUDE_MODEL, MAX_TOKENS

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

QUESTIONS = [
    ["My wifi disconnected during something important",
     "Someone ate my food in the office fridge",
     "I was given unsolicited advice",
     "I lost at something I said I didn't care about"],
    ["Coffee shop with laptop open, doing nothing productive",
     "Gym at 5am for no reason",
     "Bed. Always bed.",
     "Somewhere I wasn't invited but showed up anyway"],
    ["I did everything and I want everyone to know",
     "I contributed exactly one slide",
     "I was the vibe",
     "I disappeared and reappeared at the presentation"],
]


def generate_gym_leader_text(gym_type, q1_idx, q2_idx, q3_idx, user_name="Challenger"):
    """
    Generates opening trash talk line and badge name.

    Returns:
        opening_line : str
        badge_name   : str
    """
    answers = [
        QUESTIONS[0][q1_idx],
        QUESTIONS[1][q2_idx],
        QUESTIONS[2][q3_idx],
    ]

    prompt = f"""You are generating content for a Pokemon Gym Leader card app.

The user's gym type is: {gym_type.upper()}
Their personality answers:
- Villain origin: {answers[0]}
- Natural habitat: {answers[1]}
- Group project energy: {answers[2]}

Generate exactly two things:

1. OPENING_LINE: A short, funny, in-character Pokemon gym leader opening trash talk line (2-3 sentences max). 
It should roast the user based on their personality answers while sounding like an actual gym leader. 
Be specific to their answers, not generic. Keep it under 40 words.

2. BADGE_NAME: A funny, creative badge name that fits their personality (2-3 words max, ends in "Badge").

Respond in this exact format with no other text:
OPENING_LINE: [the line here]
BADGE_NAME: [the badge name here]"""

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}]
    )

    response = message.content[0].text.strip()
    lines = response.split('\n')

    opening_line = ''
    badge_name = ''

    for line in lines:
        if line.startswith('OPENING_LINE:'):
            opening_line = line.replace('OPENING_LINE:', '').strip()
        elif line.startswith('BADGE_NAME:'):
            badge_name = line.replace('BADGE_NAME:', '').strip()

    return opening_line, badge_name
```

---

### Task D3.2 — Pokemon Sprite Fetching
**File:** `card/fetch_sprites.py`

```python
# PokéLeader — card/fetch_sprites.py

import requests
from PIL import Image
from io import BytesIO
import numpy as np


def fetch_sprite(pokemon_name):
    """
    Fetches the front default sprite for a Pokemon from PokeAPI.
    Returns a PIL Image (RGBA) or None if fetch fails.
    """
    try:
        url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
        response = requests.get(url, timeout=5)
        data = response.json()
        sprite_url = data['sprites']['front_default']
        img_response = requests.get(sprite_url, timeout=5)
        sprite = Image.open(BytesIO(img_response.content)).convert('RGBA')
        # Upscale from 96x96 to 192x192 using nearest neighbor (keeps pixel art look)
        sprite = sprite.resize((192, 192), Image.NEAREST)
        return sprite
    except Exception as e:
        print(f'Failed to fetch {pokemon_name}: {e}')
        return None


def fetch_team_sprites(pokemon_list):
    """Fetches sprites for a list of 3 Pokemon. Returns list of PIL Images."""
    return [fetch_sprite(name) for name in pokemon_list]
```

---

### Task D3.3 — Card Assembly
**File:** `card/assemble_card.py`

```python
# PokéLeader — card/assemble_card.py

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

DRIVE_BASE   = '/content/drive/MyDrive/pokeleader'
FONT_PATH    = f'{DRIVE_BASE}/assets/fonts/pokemon_font.ttf'
BADGES_DIR   = f'{DRIVE_BASE}/assets/type_badges'

TYPE_COLORS = {
    'psychic':  '#FF6FA0',
    'fighting': '#C03028',
    'normal':   '#A8A878',
    'fairy':    '#EE99AC',
    'ghost':    '#705898',
    'electric': '#F8D030',
    'water':    '#6890F0',
    'dark':     '#705848',
}

CARD_W, CARD_H = 600, 900


def assemble_card(portrait_pil, gym_type, type_color,
                  opening_line, badge_name,
                  sprites, user_name="GYM LEADER"):
    """
    Assembles the final gym leader card.

    Args:
        portrait_pil  : PIL Image from D2 (512x512)
        gym_type      : string
        type_color    : hex color string
        opening_line  : string from Claude API
        badge_name    : string from Claude API
        sprites       : list of 3 PIL Images from PokeAPI
        user_name     : string

    Returns:
        PIL Image — the complete card (600x900)
    """
    card = Image.new('RGB', (CARD_W, CARD_H), color='#1a1a2e')
    draw = ImageDraw.Draw(card)

    # Type color header bar
    draw.rectangle([(0, 0), (CARD_W, 80)], fill=type_color)

    # Gym leader title
    try:
        font_title  = ImageFont.truetype(FONT_PATH, 18)
        font_body   = ImageFont.truetype(FONT_PATH, 11)
        font_small  = ImageFont.truetype(FONT_PATH, 9)
    except:
        font_title  = ImageFont.load_default()
        font_body   = ImageFont.load_default()
        font_small  = ImageFont.load_default()

    draw.text((CARD_W // 2, 25), user_name,
              font=font_title, fill='white', anchor='mm')
    draw.text((CARD_W // 2, 58), f"{gym_type.upper()} TYPE GYM",
              font=font_small, fill='white', anchor='mm')

    # Portrait (resize to fit card width)
    portrait_resized = portrait_pil.resize((560, 420))
    card.paste(portrait_resized, (20, 90))

    # Type color divider
    draw.rectangle([(0, 518), (CARD_W, 526)], fill=type_color)

    # Pokemon team sprites
    sprite_y = 535
    sprite_positions = [30, 210, 390]
    for i, sprite in enumerate(sprites):
        if sprite is not None:
            sprite_rgb = sprite.convert('RGBA')
            card.paste(sprite_rgb, (sprite_positions[i], sprite_y), sprite_rgb)

    # Opening line box
    draw.rectangle([(20, 740), (CARD_W - 20, 840)],
                   fill='#16213e', outline=type_color, width=2)

    # Wrap opening line text
    words = opening_line.split()
    lines_text = []
    current = ''
    for word in words:
        test = current + ' ' + word if current else word
        if len(test) > 52:
            lines_text.append(current)
            current = word
        else:
            current = test
    if current:
        lines_text.append(current)

    text_y = 755
    for line in lines_text[:3]:
        draw.text((CARD_W // 2, text_y), line,
                  font=font_small, fill='white', anchor='mm')
        text_y += 25

    # Badge name footer
    draw.rectangle([(0, 850), (CARD_W, CARD_H)], fill=type_color)
    draw.text((CARD_W // 2, 875), badge_name,
              font=font_body, fill='white', anchor='mm')

    return card
```

### Push D3 Work
```python
%%bash
git add core/generate_text.py card/fetch_sprites.py card/assemble_card.py
git commit -m "feat(d3): Claude API text gen, sprite fetching, card assembly"
git push origin dev/d3-card-ui
```

---

## Hour 5 — Integration + Gradio UI (All, led by D3)

> D3 pulls from D1 and D2 branches and wires everything together.

### Pull All Branches
```python
%%bash
git fetch origin
git checkout dev/d3-card-ui
git merge origin/dev/d1-face-type
git merge origin/dev/d2-portrait
```

### Task INT.1 — Wire Full Pipeline
**File:** `ui/app.py`

```python
# PokéLeader — ui/app.py

import sys
sys.path.append('/content/pokeleader')

import gradio as gr
import numpy as np
from PIL import Image

from core.face_check        import check_and_crop_face
from core.type_assign       import assign_type
from core.generate_portrait import generate_portrait
from core.generate_text     import generate_gym_leader_text
from card.fetch_sprites     import fetch_team_sprites
from card.assemble_card     import assemble_card
from data.gym_teams         import GYM_TEAMS

Q1_OPTIONS = [
    "My wifi disconnected during something important",
    "Someone ate my food in the office fridge",
    "I was given unsolicited advice",
    "I lost at something I said I didn't care about",
]
Q2_OPTIONS = [
    "Coffee shop with laptop open, doing nothing productive",
    "Gym at 5am for no reason",
    "Bed. Always bed.",
    "Somewhere I wasn't invited but showed up anyway",
]
Q3_OPTIONS = [
    "I did everything and I want everyone to know",
    "I contributed exactly one slide",
    "I was the vibe",
    "I disappeared and reappeared at the presentation",
]


def generate_card(user_image, user_name, q1, q2, q3):
    if user_image is None:
        return None, "Please upload a photo first."

    # Face check
    face_crop, msg = check_and_crop_face(user_image)
    if face_crop is None:
        return None, msg

    # Type assignment
    q1_idx = Q1_OPTIONS.index(q1)
    q2_idx = Q2_OPTIONS.index(q2)
    q3_idx = Q3_OPTIONS.index(q3)
    gym_type, type_color = assign_type(q1_idx, q2_idx, q3_idx)

    # Portrait generation
    portrait = generate_portrait(face_crop, gym_type)

    # Text generation
    opening_line, badge_name = generate_gym_leader_text(
        gym_type, q1_idx, q2_idx, q3_idx, user_name
    )

    # Sprite fetching
    team = GYM_TEAMS[gym_type]
    sprites = fetch_team_sprites(team)

    # Card assembly
    card = assemble_card(
        portrait_pil=portrait,
        gym_type=gym_type,
        type_color=type_color,
        opening_line=opening_line,
        badge_name=badge_name,
        sprites=sprites,
        user_name=user_name.upper() if user_name else "GYM LEADER"
    )

    status = f"You are a {gym_type.upper()} type Gym Leader. {badge_name} unlocked."
    return np.array(card), status


with gr.Blocks(title="PokéLeader") as demo:
    gr.Markdown("# PokéLeader")
    gr.Markdown("Upload your face, answer 3 questions, become a Gym Leader.")

    with gr.Row():
        with gr.Column():
            user_image = gr.Image(label="Upload your photo", type="numpy")
            user_name  = gr.Textbox(label="Your name", placeholder="Ash")
            q1 = gr.Radio(choices=Q1_OPTIONS, label="Your villain origin story")
            q2 = gr.Radio(choices=Q2_OPTIONS, label="Your natural habitat")
            q3 = gr.Radio(choices=Q3_OPTIONS, label="Your group project energy")
            submit = gr.Button("Generate My Gym Leader Card", variant="primary")

        with gr.Column():
            output_card   = gr.Image(label="Your Gym Leader Card")
            output_status = gr.Textbox(label="Result", interactive=False)

    submit.click(
        fn=generate_card,
        inputs=[user_image, user_name, q1, q2, q3],
        outputs=[output_card, output_status]
    )

if __name__ == '__main__':
    demo.launch(share=True)
```

### Launch
```python
exec(open('/content/pokeleader/ui/app.py').read())
```

### Final Push
```python
%%bash
git add ui/app.py
git commit -m "feat(d3): full Gradio UI integration complete"
git push origin dev/d3-card-ui
```

---

## End of Day — Merge to Main

Merge order: D1 first → D2 second → D3 third.

Each developer opens a Pull Request on GitHub:
1. Base: `main` ← Compare: `dev/YOUR_BRANCH`
2. Add description of what you built
3. Request review from one teammate
4. Merge after approval

---

## Handoff Summary

| Handoff | From | To | What |
|---|---|---|---|
| Hour 4 | D1 | D3 | `face_check.py` output format confirmed |
| Hour 4 | D1 | D3 | `type_assign.py` return values confirmed |
| Hour 4 | D2 | D3 | `generate_portrait()` function signature confirmed |
| Hour 5 | D3 | All | Integration branch ready for testing |

---

*PokéLeader — 1-Day Task Flow v1.0*
