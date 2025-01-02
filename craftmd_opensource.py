import sys
import pandas as pd

# Set OpenAI API Key
import openai
deployment_name = "<insert deployment name>"
openai.api_base = f"https://{deployment_name}.openai.azure.com/"
openai.api_key = "<insert API key>"

from src.utils import get_choices
from src.craftmd import craftmd_opensource, craftmd_opensource_system
from src.models import get_model_and_tokenizer

# # To download open-source models, if not already installed in your conda environment
from huggingface_hub import login
login(token = "<insert huggingface token>")

if __name__ == "__main__":
    
    model_names = ["llama2-7b", "mistral-v1", "mistral-v2"]
    dataset = pd.read_csv("./data/usmle_and_derm_dataset.csv",
                          index_col=0)
    
    cases = [(dataset.loc[idx,"case_id"], 
              dataset.loc[idx,"case_vignette"], 
              dataset.loc[idx,"category"],
              get_choices(dataset,idx)) for idx in dataset.index[start:end]]

    for model_name in model_names:
        path_dir = f"results/{model_name}"
        model, tokenizer = get_model_and_tokenizer(model_name)
                
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id 
    
        for case in cases:
            try:
                craftmd_opensource(case, path_dir, model, tokenizer, model_name)
            except Exception as e:
                print(e)
                print(f"Error in run : {case[0]}")
                continue
        