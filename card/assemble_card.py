# PokéLeader — card/assemble_card.py

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

DRIVE_BASE   = os.getcwd()
FONT_PATH    = os.path.join(DRIVE_BASE, 'assets', 'fonts', 'pokemon_font.ttf')
BADGES_DIR   = os.path.join(DRIVE_BASE, 'assets', 'type_badges')

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
    if portrait_pil:
        portrait_resized = portrait_pil.resize((560, 420))
        card.paste(portrait_resized, (20, 90))
    else:
        draw.rectangle([(20, 90), (580, 510)], fill='grey')
        draw.text((300, 300), "Portrait Pending", font=font_body, fill='white', anchor='mm')

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
