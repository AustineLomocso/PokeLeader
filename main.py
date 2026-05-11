import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("--- Starting PokeLeader ---")
    print("Note: If models are not yet downloaded, they will start downloading now.")
    print("Alternatively, wait for the background download to finish.")
    
    from ui.app import demo
    demo.launch(share=False)
