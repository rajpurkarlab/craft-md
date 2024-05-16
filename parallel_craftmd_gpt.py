import sys
import pandas as pd
from multiprocessing import Pool

# Set OpenAI API Key
import openai
deployment_name = "<insert deployment name>"
openai.api_base = f"https://{deployment_name}.openai.azure.com/"
openai.api_key = "<insert API key>"

from src.utils import get_choices
from src.craftmd import craftmd_gpt

if __name__ == "__main__":
    
    model_names = ["gpt4_1106", "gpt3_1106"]
    dataset = pd.read_csv("./data/usmle_derm_dataset.csv", index_col=0)

    # Set number of threads for parallelization
    num_cpu = 7
    
    cases = [(dataset.loc[idx,"case_id"], 
              dataset.loc[idx,"case_vignette"], 
              dataset.loc[idx,"category"],
              get_choices(dataset,idx)) for idx in dataset.index]

    for model_name in model_names:
        path_dir = f"./results/{model_name}"
        
        with Pool(num_cpu) as p:
            p.starmap(craftmd_gpt, [(x, path_dir, model_name) for x in cases])
        