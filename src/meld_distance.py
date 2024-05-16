import numpy as np
import pandas as pd
from models import call_gpt4_api
from prompts import get_meld_prompt

from Levenshtein import distance as levenshtein_distance

# Helper Functions
def split_string(s):
    midpoint = len(s) // 2
    if " " in s[midpoint:]:  # Check if there is a space in the second half
        forward_split = s[midpoint:].index(" ") + midpoint
    else:
        forward_split = len(s)  # If no space, the split point is the end of the string
    
    if " " in s[:midpoint]:  # Check if there is a space in the first half
        backward_split = s[:midpoint].rindex(" ")
    else:
        backward_split = 0  # If no space, the split point is the start of the string

    if (forward_split - midpoint) < (midpoint - backward_split):
        split_point = forward_split
    else:
        split_point = backward_split

    first_half = s[:split_point].strip()
    second_half = s[split_point:].strip()

    return first_half, second_half
                
def count_words(text):
    words = text.split()
    return len(words)

# MELD Analysis
dataset_complete = pd.read_csv("/home/shj622/rajpurkarlab/home/shj622/craft-md-v2/data/complete_dataset.csv", index_col=0)
dataset_df = dataset_complete[((dataset_complete.dataset.isin(["dermatology_public", "dermatology_private"])) | ((dataset_complete.dataset=="MedQA_USMLE") & (dataset_complete.case_vignette.str.contains("diagnosis"))))]

res_df = pd.DataFrame(columns = ["case_id", "dataset", "original_vignette", 
                                 "first_half", "second_half", "gpt4_generation", "meld_score"])

for idx, case_vignette in enumerate(tqdm(dataset_df.case_vignette)):
    case_id = dataset_df.iloc[idx, ]["case_id"]    
    dataset = dataset_df.iloc[idx,]["dataset"]
    vignette = dataset_df.iloc[idx,]["case_vignette"]
    first_half, second_half = split_string(vignette)
    
    convo = [{"role":"system", "content": get_meld_prompt(first_half, count_words(first_half))}]
    response_gpt4 = call_gpt4_api(convo)
    
    try:
        meld = levenshtein_distance(second_half, response_gpt4)/max(len(second_half), len(response_gpt4))
    except:
        meld = 0
        
    res_df.loc[idx,] = [case_id, dataset, vignette, first_half, second_half, response_gpt4, meld]
    
res_df.to_csv("./results/meld_distance_calculation.csv")