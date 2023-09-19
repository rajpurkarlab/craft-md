import json
import pandas as pd
import numpy as np
import os

from .utils import *
from .mcq_frq import *

# Task specific functions
def get_mcq_prompt(choices):
    system_prompt = " Based on the patient's symptoms described above and a list of possible diagnosis options,"
    system_prompt += " select the correct option, and give the answer as a short response. Do not explain."
    system_prompt += " \n Choices: " + choices
    
    return system_prompt

# def get_choices(case, num_choices=4):
#     list_choices = 'Choices: ' + ', '.join([str(case[f"Choice {j}"]) for j in range(1, num_choices + 1)])
#     return list_choices

# def get_all_choices(path):
#     all_choices = pd.read_csv(path, header=None)[0].tolist()
#     list_all_choices = 'Choices: ' + ', '.join(all_choices)
#     return list_all_choices

def get_diagnosis_after_physical_exam_prompt():
    system_prompt = ' Now, stop asking questions and arrive at differential diagnosis, otherwise you will be penalized.'
    system_prompt += " You must state 'Final Diagnosis:' in the beginning of your response." 
    return system_prompt

def get_physical_exam_prompt(exam):
    if not isinstance(exam, str):
        system_prompt = "There is no physical examination result available for this patient."
    else: 
        system_prompt = "Here is the examination result: " + exam
    return system_prompt

### CRAFT-MD Framework
def get_doctor_prompt(): 
    system_prompt = "You are an AI doctor specializing in dermatology."
    system_prompt += " Arrive at a differential diagnosis of a patient's medical condition."
    system_prompt += " Ask simple 1 line questions, one question at a time."
    system_prompt += " Only stop asking questions when you are 100% confident of the diagnosis, otherwise continue asking questions."
    system_prompt += " The questions should cover the age and sex of the patient, current symptoms, medical history of illness and medications, and relevant family history if necessary."
    system_prompt += " Keep your responses very minimal and brief to not confuse the patient."
    system_prompt += " When you arrive at differential diagnosis, you must state 'Final Diagnosis:' in the beginning of your response. Otherwise you will be penalized."
    return system_prompt

def get_patient_prompt(case_desc): 
    system_prompt = "You are a patient."
    system_prompt += " You do not have any medical knowledge."
    system_prompt += " Based upon questions asked, you have to describe your symptoms from the following paragraph: '"+ case_desc +"'"
    system_prompt += ". Do not break character and reveal that you are describing symptoms from a paragraph."
    system_prompt += " Do not generate any new symptoms or knowledge otherwise you will be penalized."
    system_prompt += " Do not reveal more knowledge than what the question asks."
    system_prompt += " Keep your answer to only 1 sentence."
    system_prompt += " Simplify terminology used in the given paragraph to layman language."
    return system_prompt

def process_case_withPE(case, dir_path, gpt_model, num_runs = 10):
    
    mapping = {"gpt-3.5": call_gpt3_api, "gpt-4": call_gpt4_api}
    
    idx, case_desc, exam, mcq_choices, mcq_many_choices = case

    print(f'thread for case idx: {idx} dispatched')
    
    doctor_prompt = get_doctor_prompt()
    patient_prompt = get_patient_prompt(case_desc)
    exam_prompt = get_physical_exam_prompt(exam)

    mcq_prompt = get_mcq_prompt(mcq_choices)
    mcq_all_prompt = get_mcq_prompt(mcq_many_choices)
    
    stats = {}
    j = 0

    save_path = dir_path + f'/case_{idx}.json'
    if os.path.exists(save_path): 
        with open(save_path, 'r') as f: 
            stats = json.load(f)
        while f'trial_{j}_doctor_responses' in stats: 
            j += 1
    
    while j < num_runs:
        conversation_history_doctor = [{"role": "system", "content": doctor_prompt}]
        conversation_history_patient = [{"role": "system", "content": patient_prompt}]

        while True:
            # Patient talks
            response_patient = mapping[gpt_model](conversation_history_patient, n_responses=1)

            conversation_history_doctor.append({"role":"user",
                                               "content":response_patient})
            conversation_history_patient.append({"role":"assistant",
                                               "content":response_patient})

            # Doctor talks
            response_doctor = mapping[gpt_model](conversation_history_doctor, n_responses=1)

            conversation_history_doctor.append({"role":"assistant",
                                               "content": response_doctor})
            conversation_history_patient.append({"role":"user", 
                                                "content": response_doctor})

            # Doctor arrives at a differential diagnosis
            if ("?" not in response_doctor) or ('Final Diagnosis' in response_doctor):
                break
                
        # multi-turn conversation (with physical exam) + FRQ
        prompt = exam_prompt + get_diagnosis_after_physical_exam_prompt()
        conversation_history_doctor.append({"role": "system", "content": prompt})
        response_doctor = mapping[gpt_model](conversation_history_doctor, n_responses=1)
        conversation_history_doctor.append({"role":"assistant","content": response_doctor})
        stats[f"trial_{j}_doctor_responses_with_exam"] = conversation_history_doctor
        conversation_history_doctor = conversation_history_doctor[:-2]

        # multi-turn conversation (with physical exam) + 4-choice MCQ
        prompt = exam_prompt + mcq_prompt
        conversation_history_doctor.append({"role": "system", "content": prompt})
        response_doctor = mapping[gpt_model](conversation_history_doctor, n_responses=1)
        stats[f"trial_{j}_mcq_with_exam"] = response_doctor
        conversation_history_doctor.pop()

        # multi-turn conversation (with physical exam) + many-choice MCQ
        prompt = exam_prompt + mcq_all_prompt
        conversation_history_doctor.append({"role": "system", "content": prompt})
        response_doctor = mapping[gpt_model](conversation_history_doctor, n_responses=1)
        stats[f"trial_{j}_mcq_all_with_exam"] = response_doctor
        conversation_history_doctor.pop()
 
        j += 1
        
        json.dump(stats, open(dir_path + f'/case_{idx}.json', 'w'))


def process_case_withoutPE(case, dir_path, gpt_model, num_runs = 10):
    
    mapping = {"gpt-3.5": call_gpt3_api, "gpt-4": call_gpt4_api}
    
    idx, case_desc, mcq_choices, mcq_all_choices = case

    print(f'thread for case idx: {idx} dispatched')
    
    doctor_prompt = get_doctor_prompt()
    patient_prompt = get_patient_prompt(case_desc)
        
    mcq_prompt = get_mcq_prompt_choices_only(mcq_choices)
    mcq_all_prompt = get_mcq_prompt_choices_only(mcq_all_choices)
    
    stats = {}
    j = 0

    save_path = dir_path + f'/case_{idx}.json'
    if os.path.exists(save_path): 
        with open(save_path, 'r') as f: 
            stats = json.load(f)
        while f'trial_{j}_doctor_responses' in stats: 
            j += 1
    
    while j < num_runs:
        conversation_history_doctor = [{"role": "system", "content": doctor_prompt}]
        conversation_history_patient = [{"role": "system", "content": patient_prompt}]

        while True:
            # Patient talks
            response_patient = mapping[gpt_model](conversation_history_patient, n_responses=1)

            conversation_history_doctor.append({"role":"user",
                                               "content":response_patient})
            conversation_history_patient.append({"role":"assistant",
                                               "content":response_patient})

            # Doctor talks
            response_doctor = mapping[gpt_model](conversation_history_doctor, n_responses=1)

            conversation_history_doctor.append({"role":"assistant",
                                               "content": response_doctor})
            conversation_history_patient.append({"role":"user", 
                                                "content": response_doctor})

            # Doctor arrives at a differential diagnosis
            if ("?" not in response_doctor) or ('Final Diagnosis' in response_doctor):

                # multi-turn conversation + 4-choice MCQs
                c = conversation_history_doctor[:-1]
                c.append({"role": "system", "content": mcq_prompt})

                mcq = mapping[gpt_model](c, n_responses=1)
                
                # multi-turn conversation + many-choice MCQs
                c = conversation_history_doctor[:-1]
                c.append({"role": "system", "content": mcq_all_prompt})

                mcq_all = mapping[gpt_model](c, n_responses=1)

                break

        stats[f'trial_{j}_doctor_responses'] = conversation_history_doctor[:]
        stats[f'trial_{j}_mcq'] = mcq
        stats[f'trial_{j}_mcq_all'] = mcq_all
        j += 1
        
        json.dump(stats, open(dir_path + f'/case_{idx}.json', 'w'))
