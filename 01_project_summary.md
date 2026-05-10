# PokéLeader — Pokemon Gym Leader Generator

## Project Summary

---

### What Is This?

PokéLeader is a fun, AI-powered web app where a user uploads a photo of their face, answers 3 quick personality questions, and gets back a shareable **Pokemon Gym Leader card** — fully illustrated in Pokemon art style, with their own face, gym type, Pokemon team, trash talk opening line, and badge name.

---

### The Core Idea

The app combines two separate AI pipelines:

- **Diffusion model + IP-Adapter** — generates an illustrated gym leader portrait using the user's face as a reference, preserving their likeness while applying the Pokemon art style
- **Claude API** — generates personalized text content: the gym leader's trash talk opening line and their unique badge name, based on the user's personality answers

The result is assembled into a single downloadable card using PIL post-processing.

---

### The Three Personality Questions

These are fun, relatable, and map to a gym type assignment:

**Question 1 — Your villain origin story:**
- My wifi disconnected during something important
- Someone ate my food in the office fridge
- I was given unsolicited advice
- I lost at something I said I didn't care about

**Question 2 — Your natural habitat:**
- Coffee shop with laptop open, doing nothing productive
- Gym at 5am for no reason
- Bed. Always bed.
- Somewhere I wasn't invited but showed up anyway

**Question 3 — Your energy in a group project:**
- I did everything and I want everyone to know
- I contributed exactly one slide
- I was the vibe
- I disappeared and reappeared at the presentation

---

### The Eight Gym Types

| Type | Personality Archetype |
|---|---|
| Psychic | Overpowered and knows it |
| Fighting | Unhinged discipline energy |
| Normal | Deceptively dangerous |
| Fairy | Chaotic neutral |
| Ghost | Dramatic and mysterious |
| Electric | Unpredictable energy |
| Water | Calm but will destroy you |
| Dark | Says very little |

---

### The Output Card

A single shareable image containing:

- **Illustrated portrait** — user's face in Pokemon gym leader art style
- **Gym type badge** — color coded to their assigned type
- **Pokemon team** — 3 sprites fetched from PokeAPI matching their type
- **Opening trash talk line** — Claude-generated, personalized to their answers
- **Badge name** — Claude-generated, unique to their personality
- **Gym Leader title** — their name styled as an official gym leader

---

### Why It Works

| Element | Why it lands |
|---|---|
| Personalized face | User sees themselves in the output |
| Personality-driven type | Feels accurate and fun to debate |
| Trash talk line | The funniest and most shareable part |
| Card format | Complete artifact ready to screenshot and send |
| Pokemon nostalgia | Universal reference everyone gets |

---

### What Makes It Shareable

- The trash talk opening line roasts the user based on their own answers
- The type assignment always sparks debate — people tag friends saying "this is literally you"
- The card format is designed to be screenshotted and sent directly
- Three outputs from the same photo (different type combinations) give users a reason to replay

---

### Technical Overview

```
User photo + 3 personality answers
        ↓
Face detection (MediaPipe)
        ↓
Face embedding extraction (IP-Adapter)
        ↓
Gym type assignment (rule-based logic)
        ↓
Portrait generation (Stable Diffusion + IP-Adapter + LoRA)
        ↓
Text generation (Claude API — opening line + badge name)
        ↓
Sprite fetching (PokeAPI)
        ↓
Card assembly (PIL)
        ↓
Shareable gym leader card
```

---

### Stack at a Glance

| Layer | Technology |
|---|---|
| Diffusion model | Stable Diffusion 1.5 via diffusers |
| Face preservation | IP-Adapter (h94/IP-Adapter) |
| Art style | Pokemon Trainer LoRA |
| Text generation | Claude API (claude-sonnet-4-20250514) |
| Pokemon data | PokeAPI (free, no auth) |
| UI | Gradio |
| Post-processing | PIL / Pillow |
| Runtime | Google Colab (T4 GPU) |

---

### Limitations

- IP-Adapter face resemblance drops with low quality or angled photos — front-facing selfies work best
- Output portrait is AI-generated illustration, not a photorealistic render
- Gym type assignment is rule-based, not ML — intentionally simple for the 1-day build
- Pokemon sprites are small pixel art — enlarged with nearest-neighbor for the card

---

*PokéLeader — Built with Stable Diffusion, IP-Adapter, Claude API, and PokeAPI*
