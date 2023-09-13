import pandas as pd
from multiprocessing import Pool
import os

import openai

from src.craft_md import process_case_withPE
from src.utils import *
import glob
from tqdm.auto import tqdm

import sys

if __name__ == "__main__":
    
    # Set up OpenAI API credentials for GPT-3.5
    openai_key = open("../keys/openai_key.txt", "r")
    openai.api_key = openai_key.readlines()[0].strip()

    organization_id = open("../keys/rajpurkarlab_org_id.txt", "r")
    openai.organization = organization_id.readlines()[0].strip()

    gpt_model = sys.argv[1]
    
    # Read dataset
    dataset = pd.read_csv(f"data/dataset_final.tsv", sep = "\t")
    all_choices = get_all_choices(path = './data/all_choices.txt')
    cases = [(idx, dataset.loc[idx,"case_desc"], dataset.loc[idx,"physical_exam"], get_choices(dataset,idx), all_choices) for idx in range(len(dataset))]
        
    # Run CRAFT-MD parallelly
    num_cpu = 5
    path = f'results/raw/conversation_withPE/'

    print(f'cases length: {len(cases)} num threads: {num_cpu}')
    
    with Pool(num_cpu) as p:
        p.starmap(process_case_withPE, [(case, path, gpt_model) for case in cases])
        
