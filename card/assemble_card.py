# PokéLeader — card/assemble_card.py
# Redesigned to match the dark card mockup design

from PIL import Image, ImageDraw, ImageFont
import os
import math

DRIVE_BASE = os.getcwd()
FONT_PATH  = os.path.join(DRIVE_BASE, 'assets', 'fonts', 'pokemon_font.ttf')

# Card dimensions
CARD_W, CARD_H = 600, 960

# Primary type colors — used for header, divider, footer, accents
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

# Darker shade of each type color for text on colored backgrounds
TYPE_DARK = {
    'psychic':  '#7a2040',
    'fighting': '#5a1010',
    'normal':   '#4a4a30',
    'fairy':    '#7a3a4a',
    'ghost':    '#2e1f42',
    'electric': '#5a4800',
    'water':    '#1a2a70',
    'dark':     '#2e1e10',
}

# Background fill for each type — dark tinted version of the type color
TYPE_BG = {
    'psychic':  '#1a0a12',
    'fighting': '#1a0808',
    'normal':   '#12120a',
    'fairy':    '#1a0a10',
    'ghost':    '#0d0820',
    'electric': '#1a1600',
    'water':    '#080f1a',
    'dark':     '#100808',
}

# Per-type dark panel color (sprite boxes, stat boxes, opening line box)
TYPE_PANEL = {
    'psychic':  '#2a0a20',
    'fighting': '#2a0808',
    'normal':   '#1e1e14',
    'fairy':    '#2a0a18',
    'ghost':    '#1a0d30',
    'electric': '#2a2200',
    'water':    '#0a1428',
    'dark':     '#1a0e08',
}

# Secondary type badge colors (text bg)
SECONDARY_BADGE_BG = {
    'psychic':  '#7F2050',
    'fighting': '#8B1A1A',
    'normal':   '#6A6A50',
    'fairy':    '#8B5060',
    'ghost':    '#4A3870',
    'electric': '#9A8800',
    'water':    '#2A4A9A',
    'dark':     '#4A3020',
}


def _load_fonts():
    try:
        return (
            ImageFont.truetype(FONT_PATH, 20),  # title
            ImageFont.truetype(FONT_PATH, 13),  # body
            ImageFont.truetype(FONT_PATH, 10),  # small
            ImageFont.truetype(FONT_PATH, 9),   # tiny
        )
    except Exception:
        d = ImageFont.load_default()
        return d, d, d, d


def _hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _draw_rounded_rect(draw, xy, radius, fill=None, outline=None, width=1):
    """Draw a rectangle with rounded corners."""
    x0, y0, x1, y1 = xy
    r = radius
    if fill:
        draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
        draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)
        draw.ellipse([x0, y0, x0 + 2*r, y0 + 2*r], fill=fill)
        draw.ellipse([x1 - 2*r, y0, x1, y0 + 2*r], fill=fill)
        draw.ellipse([x0, y1 - 2*r, x0 + 2*r, y1], fill=fill)
        draw.ellipse([x1 - 2*r, y1 - 2*r, x1, y1], fill=fill)
    if outline:
        draw.arc([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=outline, width=width)
        draw.arc([x1 - 2*r, y0, x1, y0 + 2*r], 270, 360, fill=outline, width=width)
        draw.arc([x0, y1 - 2*r, x0 + 2*r, y1], 90, 180, fill=outline, width=width)
        draw.arc([x1 - 2*r, y1 - 2*r, x1, y1], 0, 90, fill=outline, width=width)
        draw.line([x0 + r, y0, x1 - r, y0], fill=outline, width=width)
        draw.line([x0 + r, y1, x1 - r, y1], fill=outline, width=width)
        draw.line([x0, y0 + r, x0, y1 - r], fill=outline, width=width)
        draw.line([x1, y0 + r, x1, y1 - r], fill=outline, width=width)


def _draw_badge(draw, text, x, y, bg_color, text_color, font):
    """Draw a pill-shaped type badge."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad_x, pad_y = 10, 4
    bw = tw + pad_x * 2
    bh = th + pad_y * 2
    r = bh // 2
    _draw_rounded_rect(draw,
                       [x, y, x + bw, y + bh],
                       radius=r,
                       fill=bg_color)
    draw.text((x + pad_x, y + pad_y), text, font=font, fill=text_color)
    return bw, bh


def _wrap_text(text, max_chars):
    """Wrap text to fit within max_chars per line."""
    words = text.split()
    lines = []
    current = ''
    for word in words:
        test = current + ' ' + word if current else word
        if len(test) > max_chars:
            if current:
                lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines


def _draw_section_label(draw, x, y, text, font, type_color):
    """Draw a small uppercase section label in type color."""
    draw.text((x, y), text.upper(), font=font, fill=type_color)


def assemble_card(
    portrait_pil,
    gym_type,
    secondary_type=None,
    opening_line="...",
    badge_name="The Mystery Badge",
    sprites=None,
    user_name="GYM LEADER",
    stats=None,
):
    """
    Assembles the final gym leader card matching the dark mockup design.

    Args:
        portrait_pil   : PIL Image from SD pipeline (512x512), or None
        gym_type       : primary type string (e.g. 'ghost')
        secondary_type : optional secondary type string (e.g. 'dark')
        opening_line   : Claude-generated trash talk string
        badge_name     : Claude-generated badge name string
        sprites        : list of up to 3 PIL Images from PokeAPI, or None
        user_name      : the gym leader's name
        stats          : dict with funny stat names and values, e.g.
                         {'Anxiety': 847, 'Sleep hrs': 3, 'Pettiness': 99}
                         If None, defaults are generated from gym_type

    Returns:
        PIL Image — the assembled card (600 x 960)
    """
    if sprites is None:
        sprites = []

    if stats is None:
        # Default funny stats per type
        default_stats = {
            'psychic':  {'Overthink': 99, 'Sleep': 3,  'Smug': 88},
            'fighting': {'Reps done': 500, 'Rest days': 0, 'Ego': 77},
            'normal':   {'Naps today': 4, 'Cares given': 2, 'Vibes': 91},
            'fairy':    {'Chaos': 95, 'Glitter': 100, 'Sanity': 12},
            'ghost':    {'Anxiety': 847, 'Sleep': 3,  'Petty': 99},
            'electric': {'Impulses': 88, 'Focus': 14, 'Energy': 200},
            'water':    {'Calm': 95, 'Plans made': 12, 'Executed': 2},
            'dark':     {'Words said': 4, 'Eye rolls': 99, 'Mercy': 0},
        }
        stats = default_stats.get(gym_type, {'Power': 80, 'Speed': 70, 'Vibe': 90})

    type_color  = TYPE_COLORS.get(gym_type, '#705898')
    type_bg     = TYPE_BG.get(gym_type, '#0d0820')
    type_panel  = TYPE_PANEL.get(gym_type, '#1a0d30')
    type_dark   = TYPE_DARK.get(gym_type, '#2e1f42')

    type_color_rgb = _hex_to_rgb(type_color)

    # ── Canvas ──────────────────────────────────────────────────────────────
    card = Image.new('RGB', (CARD_W, CARD_H), color=type_bg)
    draw = ImageDraw.Draw(card)

    font_title, font_body, font_small, font_tiny = _load_fonts()

    # ── Subtle diagonal stripe texture (purely PIL, no external assets) ────
    stripe_color = tuple(min(255, c + 10) for c in _hex_to_rgb(type_bg))
    for i in range(0, CARD_W + CARD_H, 18):
        draw.line([(i, 0), (0, i)], fill=stripe_color, width=1)

    # ── Header bar ──────────────────────────────────────────────────────────
    HEADER_H = 88
    draw.rectangle([(0, 0), (CARD_W, HEADER_H)], fill=type_color)

    # Subtle header shine strip
    shine = tuple(min(255, c + 40) for c in type_color_rgb)
    draw.rectangle([(0, 0), (CARD_W, 4)], fill=shine)

    # "GYM LEADER" label — tiny, top left
    draw.text((20, 14), "GYM LEADER", font=font_tiny, fill=type_dark)

    # User name — large, centered
    draw.text((CARD_W // 2, 42), user_name.upper(),
              font=font_title, fill='white', anchor='mm')

    # Type badges — bottom right of header
    badge_x = CARD_W - 20
    sec_type_upper = secondary_type.upper() if secondary_type else None
    pri_type_upper = gym_type.upper()

    # Secondary badge (rightmost)
    if secondary_type and secondary_type != gym_type:
        sec_bg   = SECONDARY_BADGE_BG.get(secondary_type, '#444444')
        bw2, bh2 = _draw_badge(draw,
                                sec_type_upper,
                                0, 0,
                                sec_bg, '#ffffff', font_tiny)
        badge_x -= bw2
        _draw_badge(draw, sec_type_upper, badge_x, HEADER_H - bh2 - 12,
                    sec_bg, '#ffffff', font_tiny)
        badge_x -= 8

    # Primary badge
    pri_bg = type_dark
    bw1, bh1 = _draw_badge(draw, pri_type_upper, 0, 0,
                            pri_bg, '#ffffff', font_tiny)
    badge_x -= bw1
    _draw_badge(draw, pri_type_upper, badge_x, HEADER_H - bh1 - 12,
                pri_bg, '#ffffff', font_tiny)

    # ── Portrait zone ───────────────────────────────────────────────────────
    PORT_Y  = HEADER_H + 12
    PORT_H  = 390
    PORT_X  = 20
    PORT_W  = CARD_W - 40

    # Portrait background panel
    _draw_rounded_rect(draw,
                       [PORT_X, PORT_Y, PORT_X + PORT_W, PORT_Y + PORT_H],
                       radius=10,
                       fill=type_panel)

    if portrait_pil is not None:
        # Resize to fill portrait zone
        port_img = portrait_pil.resize((PORT_W, PORT_H), Image.LANCZOS)
        # Paste with rounded mask
        mask = Image.new('L', (PORT_W, PORT_H), 0)
        mask_draw = ImageDraw.Draw(mask)
        _draw_rounded_rect(mask_draw,
                           [0, 0, PORT_W, PORT_H],
                           radius=10,
                           fill=255)
        card.paste(port_img, (PORT_X, PORT_Y), mask)
    else:
        # Placeholder with icon
        cx = PORT_X + PORT_W // 2
        cy = PORT_Y + PORT_H // 2
        r  = 40
        dim_color = tuple(min(255, c + 30) for c in _hex_to_rgb(type_panel))
        draw.ellipse([cx - r, cy - r - 15, cx + r, cy + r - 15],
                     outline=dim_color, width=2)
        draw.arc([cx - r//2, cy + r - 15, cx + r//2, cy + 2*r - 15],
                 0, 180, fill=dim_color, width=2)
        draw.text((cx, cy + r + 16), "Portrait pending",
                  font=font_tiny, fill=dim_color, anchor='mm')

    # Bottom vignette fade on portrait
    fade_h = 60
    for i in range(fade_h):
        alpha = int(200 * (i / fade_h))
        fade_color = tuple(list(_hex_to_rgb(type_bg)) )
        row_y = PORT_Y + PORT_H - fade_h + i
        if 0 <= row_y < CARD_H:
            draw.rectangle([(PORT_X, row_y), (PORT_X + PORT_W, row_y + 1)],
                           fill=(*_hex_to_rgb(type_bg), alpha))

    # ── Type color divider ───────────────────────────────────────────────────
    DIV_Y = PORT_Y + PORT_H + 14
    draw.rectangle([(0, DIV_Y), (CARD_W, DIV_Y + 3)], fill=type_color)

    # ── Stats row ────────────────────────────────────────────────────────────
    STATS_Y  = DIV_Y + 14
    stat_keys = list(stats.keys())[:3]
    n_stats   = len(stat_keys)
    stat_w    = (CARD_W - 40 - (n_stats - 1) * 10) // n_stats

    for i, key in enumerate(stat_keys):
        sx = 20 + i * (stat_w + 10)
        sy = STATS_Y
        sh = 62
        _draw_rounded_rect(draw,
                           [sx, sy, sx + stat_w, sy + sh],
                           radius=8,
                           fill=type_panel,
                           outline=type_color + '60',
                           width=1)
        val_str = str(stats[key])
        draw.text((sx + stat_w // 2, sy + 20), val_str,
                  font=font_title, fill='white', anchor='mm')
        draw.text((sx + stat_w // 2, sy + 46), key.upper(),
                  font=font_tiny, fill=type_color, anchor='mm')

    # ── Pokemon team ────────────────────────────────────────────────────────
    TEAM_Y = STATS_Y + 76
    _draw_section_label(draw, 20, TEAM_Y, "Pokemon team", font_tiny, type_color)

    SPRITE_Y  = TEAM_Y + 18
    SPRITE_BOX_W = (CARD_W - 40 - 20) // 3
    SPRITE_BOX_H = 110

    for i in range(3):
        bx = 20 + i * (SPRITE_BOX_W + 10)
        by = SPRITE_Y
        _draw_rounded_rect(draw,
                           [bx, by, bx + SPRITE_BOX_W, by + SPRITE_BOX_H],
                           radius=8,
                           fill=type_panel,
                           outline=type_color + '50',
                           width=1)
        if i < len(sprites) and sprites[i] is not None:
            sprite = sprites[i].convert('RGBA')
            sprite = sprite.resize((80, 80), Image.NEAREST)
            sx = bx + (SPRITE_BOX_W - 80) // 2
            sy = by + 8
            card.paste(sprite, (sx, sy), sprite)
            # Pokemon name placeholder under sprite
            draw.text((bx + SPRITE_BOX_W // 2, by + SPRITE_BOX_H - 14),
                      f"pokemon {i+1}",
                      font=font_tiny, fill=type_color, anchor='mm')
        else:
            draw.text((bx + SPRITE_BOX_W // 2, by + SPRITE_BOX_H // 2),
                      "?",
                      font=font_body, fill=type_color, anchor='mm')

    # ── Opening line box ────────────────────────────────────────────────────
    LINE_Y = SPRITE_Y + SPRITE_BOX_H + 16
    _draw_section_label(draw, 20, LINE_Y, "Opening line", font_tiny, type_color)

    BOX_Y  = LINE_Y + 18
    BOX_H  = 100
    _draw_rounded_rect(draw,
                       [20, BOX_Y, CARD_W - 20, BOX_Y + BOX_H],
                       radius=8,
                       fill=type_panel,
                       outline=type_color,
                       width=2)

    # Quotation mark accent
    draw.text((34, BOX_Y + 6), "\u201c", font=font_title, fill=type_color)

    # Wrapped opening line text
    wrapped = _wrap_text(opening_line, max_chars=56)
    text_y  = BOX_Y + 16
    for line in wrapped[:3]:
        draw.text((CARD_W // 2, text_y), line,
                  font=font_small, fill='white', anchor='mm')
        text_y += 22

    # ── Footer — badge name ──────────────────────────────────────────────────
    FOOT_Y = BOX_Y + BOX_H + 16
    draw.rectangle([(0, FOOT_Y), (CARD_W, CARD_H)], fill=type_color)

    # Shine strip
    draw.rectangle([(0, FOOT_Y), (CARD_W, FOOT_Y + 3)], fill=shine)

    # Star badge icon (left)
    STAR_CX, STAR_CY = 40, FOOT_Y + (CARD_H - FOOT_Y) // 2
    STAR_R  = 14
    star_points = []
    for j in range(10):
        angle  = math.pi / 2 + j * 2 * math.pi / 10
        radius = STAR_R if j % 2 == 0 else STAR_R * 0.45
        star_points.append((
            STAR_CX + radius * math.cos(angle),
            STAR_CY - radius * math.sin(angle),
        ))
    draw.polygon(star_points, fill='white', outline=type_dark)

    # Badge name text — centered
    draw.text((CARD_W // 2, FOOT_Y + (CARD_H - FOOT_Y) // 2),
              badge_name.upper(),
              font=font_body, fill='white', anchor='mm')

    # Star badge icon (right, mirrored)
    STAR_CX2 = CARD_W - 40
    star_points2 = []
    for j in range(10):
        angle  = math.pi / 2 + j * 2 * math.pi / 10
        radius = STAR_R if j % 2 == 0 else STAR_R * 0.45
        star_points2.append((
            STAR_CX2 + radius * math.cos(angle),
            STAR_CY  - radius * math.sin(angle),
        ))
    draw.polygon(star_points2, fill='white', outline=type_dark)

    return card


# ── Quick test (run directly) ────────────────────────────────────────────────
if __name__ == '__main__':
    from card.fetch_sprites import fetch_team_sprites

    test_sprites = fetch_team_sprites(['gengar', 'mimikyu', 'umbreon'])

    card = assemble_card(
        portrait_pil   = None,
        gym_type       = 'ghost',
        secondary_type = 'dark',
        opening_line   = "I knew you were coming. I also knew you'd lose. I've already planned what I'm having for dinner after this. It's going to be nice.",
        badge_name     = "The Overthink Badge",
        sprites        = test_sprites,
        user_name      = "Alex",
        stats          = {'Anxiety': 847, 'Sleep hrs': 3, 'Pettiness': 99},
    )

    out_path = os.path.join(DRIVE_BASE, 'test_card_output.png')
    card.save(out_path)
    print(f"Card saved to {out_path}")
    card.show()