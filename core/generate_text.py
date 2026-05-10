# PokéLeader — core/generate_text.py

import anthropic
import os
from train.config import CLAUDE_MODEL, MAX_TOKENS

# Initialize client only if API key is present
def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)

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
    """
    client = get_client()
    if not client:
        return "I'm ready to battle! (API Key missing)", "Placeholder Badge"

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
