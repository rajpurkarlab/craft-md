##### GPT Models #####
import time
from openai import AzureOpenAI
import openai

client = AzureOpenAI(api_version="2023-05-15",
                     api_key = openai.api_key,
                     azure_endpoint=openai.api_base)

def call_gpt4_api(convo, max_tokens=150, depth_limit=0):
    try:
        response = client.chat.completions.create(
                        model="gpt41106",
                        messages = convo,
                        max_tokens = max_tokens)

        completion = response.model_dump()["choices"][0]["message"]["content"]
        return completion

    except Exception as e:
        if "content" in str(e):
            print(e)
            return None
        else:
            if "retry" in str(e):
                if depth_limit <= 10:
                    time.sleep(2)
                    return call_gpt4_api(convo, max_tokens, depth_limit+1)
                else:
                    return None
                
            else:
                print(e)
                return call_gpt4_api(convo, max_tokens, depth_limit+1)
            
def call_gpt3_api(convo, max_tokens=150, depth_limit=0):
    try:
        response = client.chat.completions.create(
                        model="gpt31106",
                        messages = convo,
                        max_tokens = max_tokens)

        completion = response.model_dump()["choices"][0]["message"]['content']
        return completion

    except Exception as e:
        if "content" in str(e):
            print(e)
            return None
        else:
            if "retry" in str(e):
                if depth_limit <= 10:
                    time.sleep(2)
                    return call_gpt3_api(convo, max_tokens, depth_limit+1)
                else:
                    return None
                
            else:
                print(e)
                return call_gpt3_api(convo, max_tokens, depth_limit+1)
            
            
##### Open-Source Models #####

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def get_model_and_tokenizer(model_name:str):
    allowed_model_names = ["mistral-v1", "mistral-v2", "llama2-7b"]
    if model_name not in allowed_model_names:
        raise ValueError(f"model_name must be one of {allowed_model_names}, got '{model_name}' instead.")

    dmap = {"mistral-v1":"mistralai/Mistral-7B-Instruct-v0.1",
            "mistral-v2":"mistralai/Mistral-7B-Instruct-v0.2",
            "llama2-7b": "meta-llama/Llama-2-7b-chat-hf"}

    model = AutoModelForCausalLM.from_pretrained(dmap[model_name], device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(dmap[model_name])

    return model, tokenizer

def call_open_llm(model, tokenizer, convo):

    inputs = tokenizer.apply_chat_template(convo, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=250)
    new_tokens = outputs[0][len(inputs[0]):]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return decoded

##### Multimodal LLMs #####
import requests

def call_gpt4v_api(convo, deployment_name, max_tokens=150, depth_limit=0):
    
    headers = {"Content-Type": "application/json", "api-key": openai.api_key}
    payload = {"messages": convo, "max_tokens": max_tokens, "stream": False}

    response = requests.post(f"https://{deployment_name}.openai.azure.com/openai/deployments/gpt4visionpreview/chat/completions?api-version=2023-12-01-preview", 
                         headers=headers, json=payload).json()

    try:
        completion = response["choices"][0]["message"]["content"]
        return completion
    
    except:
        if response["error"]["code"]=="429":
            time.sleep(5)
            return call_gpt4v_api(convo, max_tokens, depth_limit+1)
        else:
            if depth_limit<=10:
                return call_gpt4v_api(convo, max_tokens, depth_limit+1)
            else:
                return None  