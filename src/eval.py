# Code for grader-AI and fuzzy string match
import pandas as pd
from fuzzywuzzy import fuzz
from .utils import *

# helper functions to get correct
def get_vignette_answer(case, dataset_df):
    idx = dataset_df[dataset_df.case_id==case].index[0]
    return dataset_df.loc[idx,["answer_name"]].values[0]

def get_all_answers(case, dataset_df):
    idx = dataset_df[dataset_df.case_id==case].index[0]
    return dataset_df.loc[idx,["answer_name"]+[f"other_answer_{i+1}" for i in range(5)]].dropna().values.tolist()

# Fuzzy String Match
def eval_fuzzy_string_match(token1, token2):
    def preprocess(token):
        return token.lower().replace('final ', '').replace('diagnosis','').replace('symptom', '')
    return int(fuzz.token_sort_ratio(preprocess(token1), preprocess(token2)) > 75)

def eval_fuzzy_string_match_multiple(response, answers):
    evals = []
    for ans in answers:
        evals.append(eval_fuzzy_string_match(response, ans))    
    return evals

# Grader-AI

# extract the diagnosis name from doctor-AI agent's response
def extract_diagnosis_name_prompt(diagnosis_para):
    system_prompt = "Identify and return the dermatology diagnosis name from the given paragraph."
    system_prompt += " If there are more than one diagnoses present, return 'Multiple'."
    system_prompt += " If there are no diagnoses present, then return 'None'."
    system_prompt += " Do not explain."
    system_prompt += "\nParagraph :" + diagnosis_para
    return system_prompt
    
def extract_diagnosis_name(diagnosis_para, gpt_model):
    convo = [{"role":"system","content":extract_diagnosis_name_prompt(diagnosis_para)}]
    if gpt_model=="gpt-3.5":
        response = call_gpt3_api(convo, n_responses=1)
    elif gpt_model=="gpt-4":
        response = call_gpt4_api(convo, n_responses=1)
    else:
        raise ValueError("Use the correct gpt model options: gpt-3.5 or gpt-4.")
    return response

# evaluate diagnosis == correct answer
def helper_eval_responses(res):
    if "yes" in res.lower():
        return 1
    elif "no" in res.lower():
        return 0
    else:
        return -1
        
def get_prompt_short_responses(choice1, choice2):
    system_prompt = "Are the two dermatology conditions the same or have synonymous names of diseases? "
    system_prompt += "Respond with a yes/no. Do not explain."
    system_prompt += "\nExample: Choice 1: eczema Choice 2: eczema They are the same, so return yes. "
    system_prompt += "\nExample: Choice 1: wart Choice 2: wart They are the same, so return yes. "
    
    system_prompt += "\nExample: Choice 1: eczema Choice 2: onychomycosis They are different, so return no. "
    system_prompt += "\nExample: Choice 1: wart Choice 2: alopecia areata They are different, so return no. "
    
    system_prompt += "\nExample: Choice 1: eczema Choice 2: atopic dermatitis They are synonymous, so return yes. "
    system_prompt += "\nExample: Choice 1: benign nevus Choice 2: mole They are synonymous, so return yes. "
    system_prompt += "\nExample: Choice 1: toe nail fungus Choice 2: onychomycosis They are synonymous, so return yes. "
    system_prompt += "\nExample: Choice 1: wart Choice 2: verruca vulgaris They are synonymous, so return yes. "

    system_prompt += "\n\nChoice 1: "+choice1
    system_prompt += "\nChoice 2: "+choice2
    return system_prompt
        
def eval_short_responses(response, ans, gpt_model):
    prompt = get_prompt_short_responses(response,ans)
    convo = [{"role":"system","content":prompt}]
    if gpt_model == "gpt-3.5":
        res = call_gpt3_api(convo, n_responses=1, max_tokens=50)
    elif gpt_model == "gpt-4":
        res = call_gpt4_api(convo, n_responses=1, max_tokens=50)
    elif gpt_model == "gpt-3.5-davinci":
        res = call_gpt3_davinci_api(prompt, n_responses=1, max_tokens=50)    
    else:
        raise ValueError("Select an appropriate gpt model.")
        
    temp = helper_eval_responses(res)
    if temp==-1:
        return eval_short_responses(response, ans, gpt_model)
    else:
        return temp
    
def eval_short_responses_multiple(response, answers, gpt_model):
    evals = []
    for ans in answers:
        evals.append(eval_short_responses(response, ans, gpt_model))    
    return evals

def eval_experiment(res_dict, case_id, exp_keys, method, full_dataset, n_trials=10):    
    eval_functions = {"mcq_4":{"stringmatch":eval_fuzzy_string_match, "autoeval":eval_short_responses},
                      "mcq_many":{"stringmatch":eval_fuzzy_string_match_multiple, "autoeval":eval_short_responses_multiple},
                      "frq":{"stringmatch":eval_fuzzy_string_match_multiple, "autoeval":eval_short_responses_multiple},
                     }
    
    ans_all = get_all_answers(case_id, full_dataset)

    for k in exp_keys:
        res_dict[k][f"{method}_raw"] = []
        res_dict[k][f"{method}_raw"] = []
        res_dict[k][f"{method}_raw"] = []

    counts = [0]*len(exp_keys)

    for i in range(n_trials):
        for k in range(len(exp_keys)):
            if exp_keys[k]=="mcq_4":
                ans = ans_all[0]
            else:
                ans = ans_all
                
            if method=="stringmatch":
                temp = eval_functions[exp_keys[k]][method](res_dict[exp_keys[k]]["responses"][i], ans)
            elif method=="autoeval":
                temp = eval_functions[exp_keys[k]][method](res_dict[exp_keys[k]]["responses"][i], ans, "gpt-4")
            res_dict[exp_keys[k]][f"{method}_raw"].append(temp)

    for k in exp_keys:
        if k=="mcq_4":
            res_dict[k][f"{method}_accuracy"] = sum(res_dict[k][f"{method}_raw"])/n_trials
        else:
            res_dict[k][f"{method}_accuracy"] = (sum([x[0] for x in res_dict[k][f"{method}_raw"]])/n_trials,
                                                 sum([max(x) for x in res_dict[k][f"{method}_raw"]])/n_trials)
        
    return res_dict