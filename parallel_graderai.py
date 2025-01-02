import sys
import pandas as pd
from multiprocessing import Pool

# Set OpenAI API Key
import openai
deployment_name = "<insert deployment name>"
openai.api_base = f"https://{deployment_name}.openai.azure.com/"
openai.api_key = "<insert API key>"

from src.graderai_eval import graderai_evaluation

if __name__ == "__main__":

    model_names = ["gpt4_1106", "gpt3_1106", "llama2-7b", "mistral-v1", "mistral-v2"]
    num_cpu = 7

    dataset = pd.read_csv("data/usmle_and_derm_dataset.csv", index_col=0)
    cases = [dataset.loc[idx,"case_id"] for idx in dataset.index]
    
    experiment_names = ["vignette_frq", "multiturn_frq", "singleturn_frq", "summarized_frq"]
    
    for model_name in model_names:
        path_dir = f"results/{model_name}"
    
        with Pool(num_cpu) as p:    
            p.starmap(graderai_evaluation, [(case_id, dataset, path_dir, experiment_names) for case_id in cases])