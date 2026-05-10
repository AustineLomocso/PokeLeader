import gradio as gr

def process(image, q1, q2, q3):
    return "Generated Card"

iface = gr.Interface(fn=process, inputs=["image", "text", "text", "text"], outputs="text")
if __name__ == "__main__":
    iface.launch()
