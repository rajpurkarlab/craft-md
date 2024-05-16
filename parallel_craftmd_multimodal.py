import pandas as pd
from multiprocessing import Pool

# Set OpenAI API Key
import openai
deployment_name = "<insert deployment name>"
openai.api_base = f"https://{deployment_name}.openai.azure.com/"
openai.api_key = "<insert API key>"

from src.utils import get_choices
from src.craftmd import craftmd_multimodal

if __name__ == "__main__":
    model_name = "gpt4v"
    dataset = pd.read_csv("./data/nejmai_dataset.csv", index_col=0)
    
    # Set number of threads for parallelization    
    num_cpu = 7

    cases = [(dataset.loc[idx,"case_id"],
              dataset.loc[idx,"case_vignette"],
              get_choices(dataset,idx)) for idx in dataset.index]

    path_dir = f"./results/{model_name}"
    img_dir = f"./data/nejmai_imgs/"
    
    with Pool(num_cpu) as p:
        p.starmap(craftmd_multimodal, [(x, img_dir, path_dir, deployment_name) for x in cases])