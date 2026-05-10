# PokéLeader — ui/app.py

import sys
import os
import gradio as gr
import numpy as np
from PIL import Image

# Ensure the project root is in path
sys.path.append(os.getcwd())

from core.face_check        import check_and_crop_face
from core.type_assign       import assign_type
# Importing with check to avoid immediate error if models not loaded
try:
    from core.generate_portrait import generate_portrait
except ImportError:
    def generate_portrait(x, y): return None

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
    try:
        q1_idx = Q1_OPTIONS.index(q1)
        q2_idx = Q2_OPTIONS.index(q2)
        q3_idx = Q3_OPTIONS.index(q3)
    except ValueError:
        return None, "Please answer all personality questions."
        
    gym_type, type_color = assign_type(q1_idx, q2_idx, q3_idx)

    # Portrait generation (In a real app, this would use the SD model)
    try:
        portrait = generate_portrait(face_crop, gym_type)
    except Exception as e:
        print(f"Portrait generation failed: {e}")
        portrait = None

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
    demo.launch()
