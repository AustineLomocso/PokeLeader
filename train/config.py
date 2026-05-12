# PokéLeader — train/config.py

IMAGE_SIZE      = 512          # SD 1.5 native resolution
NUM_STEPS       = 30            # Reduced for testing
GUIDANCE_SCALE  = 7.5          # CFG scale
LORA_WEIGHT     = 0.8          # Pokemon art style LoRA strength
IP_ADAPTER_SCALE = 0.7         # Face preservation strength (0=ignore face, 1=exact face)
CLAUDE_MODEL    = "claude-sonnet-4-20250514"
GEMINI_MODEL    = "gemini-2.0-flash"
MAX_TOKENS      = 300
