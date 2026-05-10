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
