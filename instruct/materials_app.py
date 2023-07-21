import gradio as gr
import os
from transformers import pipeline
import torch

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ["GRADIO_TEMP_DIR"] = "tmp/"

text_gen_pipeline = pipeline('text-generation', model='/export/isr-nas1/users/rmanuvin_aclgpu05/llama_quantized/falcon-7b-instruct/falcon-7b-instruct', trust_remote_code=True)
def branch_rec(input_text):
    output_text = text_gen_pipeline(input_text)[0]['generated_text']
    return "test {}".format(output_text)


with gr.Blocks(theme=gr.themes.Default(spacing_size=gr.themes.sizes.spacing_sm, radius_size=gr.themes.sizes.radius_none)) as demo:
    with gr.Row():        
        input_text = gr.Textbox(label="input_text")
        go_button = gr.Button("go")            

    with gr.Row():        
        output_text = gr.Textbox(label="output_text")

    go_button.click(fn=branch_rec, inputs=[input_text], outputs = [output_text])
                                    

demo.launch(server_name="0.0.0.0")