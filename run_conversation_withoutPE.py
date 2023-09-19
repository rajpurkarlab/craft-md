import pandas as pd
from multiprocessing import Pool
import os
import sys

import openai

from src.craft_md import process_case_withoutPE
from src.mcq_frq import get_choices, get_all_choices
from src.utils import *
import glob

if __name__ == "__main__":
    
    # Set up OpenAI API credentials
    openai_key = open("../keys/openai_key.txt", "r")
    openai.api_key = openai_key.readlines()[0].strip()

    organization_id = open("../keys/rajpurkarlab_org_id.txt", "r")
    openai.organization = organization_id.readlines()[0].strip()
    
    gpt_model = sys.argv[1]

    # Read dataset
    dataset = pd.read_csv("./data/dataset_final.tsv", sep = "\t")
    all_choices = get_all_choices(path = './data/all_choices.txt')
    
    cases = [(dataset.loc[idx,"case_id"], 
             dataset.loc[idx,"case_desc"], 
             get_choices(dataset,idx), 
             all_choices) for idx in range(dataset.shape[0])]
    
    num_trials = 10
    
    # Run CRAFT-MD parallelly 
    num_cpu = 5
    path = "./results/conversations_raw/casewise_withoutPE/"
    
    print(f'cases length: {len(cases)} num threads: {num_cpu}')
    
    with Pool(num_cpu) as p:
        p.starmap(process_case_withoutPE, [(case, path, get_model, num_trials) for case in cases])
        
