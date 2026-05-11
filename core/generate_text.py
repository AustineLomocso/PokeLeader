# PokéLeader — core/generate_text.py

import google.generativeai as genai
import os
from dotenv import load_dotenv
from train.config import GEMINI_MODEL, MAX_TOKENS

# Load .env file if it exists
load_dotenv()

# Initialize Gemini only if API key is present
def get_gemini_model():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)

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


FALLBACK_LINES = {
    'psychic': "Your thoughts are like an open book, and frankly, I'm bored. My Pokemon will shut that book for good!",
    'fighting': "You call that a battle stance? My team has more discipline in their sleep than you have in your whole body!",
    'normal': "Don't let the 'Normal' label fool you. We're consistently better than you in every way!",
    'fairy': "Oh, look at you! So brave, so small, so about to be crushed by something sparkly and adorable!",
    'ghost': "The shadows are whispering about your defeat. I'm just here to make it official. Boo!",
    'electric': "You look like you need a jumpstart! My team is about to provide a very shocking reality check!",
    'water': "You're out of your depth, Challenger. Let the waves of your own failure wash over you!",
    'dark': "The night is long, and your journey ends here. No need for words, the result will speak for itself.",
}

FALLBACK_BADGES = {
    'psychic': "Mind-Bender Badge",
    'fighting': "Iron-Fist Badge",
    'normal': "Plain-Truth Badge",
    'fairy': "Glitter-Bomb Badge",
    'ghost': "Spirit-Walker Badge",
    'electric': "Short-Circuit Badge",
    'water': "High-Tide Badge",
    'dark': "Midnight-Eclipse Badge",
}

def generate_gym_leader_text(gym_type, q1_idx, q2_idx, q3_idx, user_name="Challenger"):
    """
    Generates opening trash talk line and badge name using Gemini.
    """
    model = get_gemini_model()
    
    answers = [
        QUESTIONS[0][q1_idx],
        QUESTIONS[1][q2_idx],
        QUESTIONS[2][q3_idx],
    ]

    if not model:
        # Fallback if no API key
        line = FALLBACK_LINES.get(gym_type, "I'm ready to battle!")
        badge = FALLBACK_BADGES.get(gym_type, "Gym Badge")
        return line, badge

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

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=MAX_TOKENS,
                temperature=0.7,
            )
        )
        
        text = response.text.strip()
        lines = text.split('\n')

        opening_line = ''
        badge_name = ''

        for line in lines:
            if line.startswith('OPENING_LINE:'):
                opening_line = line.replace('OPENING_LINE:', '').strip()
            elif line.startswith('BADGE_NAME:'):
                badge_name = line.replace('BADGE_NAME:', '').strip()
        
        # If Gemini fails to follow format perfectly
        if not opening_line: opening_line = FALLBACK_LINES.get(gym_type)
        if not badge_name: badge_name = FALLBACK_BADGES.get(gym_type)

        return opening_line, badge_name

    except Exception as e:
        print(f"Gemini API error: {e}")
        return FALLBACK_LINES.get(gym_type), FALLBACK_BADGES.get(gym_type)
