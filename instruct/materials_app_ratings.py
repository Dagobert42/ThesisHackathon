import gradio as gr
import os
from transformers import pipeline
import torch

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ["GRADIO_TEMP_DIR"] = "tmp/"

# TODO: Try MPT
text_gen_pipeline = pipeline('text-generation', model='gpt2', max_new_tokens=100)
def branch_rec(input_text):
    output_text = text_gen_pipeline(input_text)[0]['generated_text']
    return "{}".format(output_text)

def save_response(input_text, output_text, ratings):
    with open('responses_from_people.txt', 'a') as f:
        f.write("{}\n".format({"input_text": input_text, "output_text": output_text, "ratings": ratings}))
    return "saved {}".format(input_text)

with gr.Blocks(theme=gr.themes.Default(spacing_size=gr.themes.sizes.spacing_sm, radius_size=gr.themes.sizes.radius_none)) as demo:
    with gr.Row():        
        input_text = gr.Textbox(label="input_text")
        go_button = gr.Button("go")            

    with gr.Row():        
        output_text = gr.Textbox(label="output_text")
    
    with gr.Row():
        gen_rating = gr.Radio(["Bad", "Ok", "Good"], label="gen_rating", info="Rate the output text")
        thanks = gr.Textbox(label="O")        
        save_rating_button = gr.Button("save_rating")            

    go_button.click(fn=branch_rec, inputs=[input_text], outputs = [output_text])
    save_rating_button.click(fn=save_response, inputs=[input_text, output_text, gen_rating], outputs = [thanks])
                                    

demo.launch(server_name="0.0.0.0")