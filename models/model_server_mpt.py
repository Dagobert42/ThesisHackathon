"""
A model worker executes the model.
"""
import argparse
import asyncio
import dataclasses
import logging
import os
import json
import time
from typing import List, Union
import threading
import uuid
import transformers
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import uvicorn
from functools import partial

WORKER_HEART_BEAT_INTERVAL = 15

worker_id = str(uuid.uuid4())[:6]
global_counter = 0

def load_model(model_path):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding = True,
            truncation = True, max_length = 512, return_tensors = 'pt')
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    return tokenizer, pipeline


class ModelWorker:
    def __init__(self, worker_addr, worker_id, model_path):
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_name = model_path
        self.tokenizer, self.pipeline = load_model(model_path)
        
    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, pipeline = self.tokenizer, self.pipeline

        prompt = params["prompt"]
        ori_prompt = prompt
        
        l_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512,'return_tensors':'pt'}
        sequences = pipeline(
                prompt,
                max_length=512,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                #**tokenizer_kwargs
            )
        gen_text = sequences[0]['generated_text']
        ret = {
                "text": gen_text,
                "error_code": 0,
            }
        yield json.dumps(ret).encode() + b"\0"
  

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": '{}'.format(e),
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": '{}'.format(e),
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()

@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global global_counter
    global_counter += 1
    print (global_counter)
    params = await request.json()    
    generator = worker.generate_stream_gate(params)

    return StreamingResponse(generator)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()

'''
python model_server_falcon.py --host 0.0.0.0 --port 30001 --worker http://localhost:30000 --model-path /export/isr-nas1/users/rmanuvin_aclgpu05/llama_quantized/falcon-7b-instruct/falcon-7b-instruct
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")    
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)    
    args = parser.parse_args()
    
    worker = ModelWorker(
                         args.worker_address,
                         worker_id,
                         args.model_path)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
