import re
from .models import call_gpt3_api, call_gpt4_api
from .prompts import convert_to_summarized_prompt, get_extract_diagnosis_name_prompt, get_diagnosis_evaluation_prompt

import base64
import requests
from mimetypes import guess_type

### General helper functions
def get_choices(cases, case_idx):
    cols = cases.columns.str.contains("choice")
    list_choices = ", ".join(cases.loc[case_idx, cols])
    return list_choices

def get_case_without_question(case):
    regex = r'([A-z][^.!?]*[.!?]*"?)'
    last_sentence = None
    if "\n" in case:
        temp = case.split("\n")[-1]
        for sentence in re.findall(regex, temp):
            if "?" in sentence:
                last_sentence = sentence
                break
    else:
        for sentence in re.findall(regex, case):
            if "?" in sentence:
                last_sentence = sentence
                break            
    if last_sentence is not None:
        return case.replace(last_sentence, ""), last_sentence.replace("of the following","")
    else:
        return case, "What is the final diagnosis?"
    
def get_correct_answer(dataset, case_id):
    return dataset.loc[dataset.case_id==case_id, "answer"].values[0].lower()
    
### Summarized conversation helper functions
def get_patient_responses(convo):
    temp = ""
    for c in convo:
        if c["role"]=="user":
            temp += c["content"]
            temp += " "
    return temp 

def convert_to_summarized(pat_dialogues):
    convo = [{"role":"system", "content":convert_to_summarized_prompt(pat_dialogues)}]
    response = call_gpt3_api(convo, max_tokens=500)
    return response
    
### Grader AI agent evaluation helper function
def extract_diagnosis_name(diagnosis_para):
    convo = [{"role":"system","content":get_extract_diagnosis_name_prompt(diagnosis_para)}]
    response = call_gpt4_api(convo)
    return response

# evaluate diagnosis == correct answer
def helper_eval_responses(res):
    if "yes" in res.lower():
        return 1
    elif "no" in res.lower():
        return 0
    else:
        return -1
    
def diagnosis_evaluation(correct_ans, clinical_llm_response, depth=0):
    clinical_llm_ans = extract_diagnosis_name(clinical_llm_response).lower()
    convo = [{"role":"system","content":get_diagnosis_evaluation_prompt(correct_ans, clinical_llm_ans)}]
    response = call_gpt4_api(convo)
    temp = helper_eval_responses(response)
    if temp == -1 and depth<=10:
        return diagnosis_evaluation(correct_ans, clinical_llm_response, depth+1)
    else:
        return temp, clinical_llm_ans
    
##### Multimodal LLM image conversion helper function
def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    return f"data:{mime_type};base64,{base64_encoded_data}"