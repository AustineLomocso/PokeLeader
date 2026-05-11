import torch
from core.generate_portrait import generate_portrait
import numpy as np
from PIL import Image

def test_gen():
    print("Starting generation test...")
    # Dummy face image (256x256)
    dummy_face = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    try:
        portrait = generate_portrait(dummy_face, 'ghost')
        if portrait:
            print("Generation successful!")
            portrait.save("test_output_portrait.png")
        else:
            print("Generation returned None.")
    except Exception as e:
        print(f"Generation failed with error: {e}")

if __name__ == "__main__":
    test_gen()
