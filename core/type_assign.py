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
