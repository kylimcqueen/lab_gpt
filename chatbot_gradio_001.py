import gradio as gr
import openai

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height = 300)

demo.launch()