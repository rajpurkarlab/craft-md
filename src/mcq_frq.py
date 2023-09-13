import openai
import pandas as pd

def get_patient_description(cases, case_idx):
    patient_description = "Patient's symptoms: " + cases['case_desc'][case_idx]
    return patient_description

def get_choices(cases, case_idx, num_choices=4):
    start = 3
    list_choices = 'Choices: ' + ', '.join(list(cases.iloc[case_idx,start:start+num_choices]))
    return list_choices

def get_all_choices(path = "./data/all_choices.txt"):
    all_choices = pd.read_csv(path, header=None)[0].tolist()
    list_all_choices = 'Choices: ' + ', '.join(all_choices)
    return list_all_choices

def get_correct_choice(cases, case_idx):
    return cases.loc[case_idx, "Choice " + str(cases.loc[case_idx,"Answer"])]

def get_mcq_prompt(desc, choices):
    # Prompt used for Vignette + MCQ and summarized conversation + MCQ
    system_prompt = "You are an AI doctor specializing in dermatology."
    system_prompt += " You are given the patient's symptoms and a list of possible diagnosis options."
    system_prompt += " Only one of the choices is correct."
    system_prompt += " Select the correct option, and give the answer as a short response. Do not explain."
    system_prompt += "\nSymptoms: "+ desc
    system_prompt += "\nChoices: "+ choices
    return system_prompt

def get_mcq_withPE_prompt(desc, age, sex, pe, choices):
    # Prompt used for Vignette + MCQ and summarized conversation + MCQ
    system_prompt = "You are an AI doctor specializing in dermatology."
    system_prompt += " You are given the patient's symptoms and a list of possible diagnosis options."
    system_prompt += " Only one of the choices is correct."
    system_prompt += " Select the correct option, and give the answer as a short response. Do not explain."
    system_prompt += "\nSymptoms: "+ desc
    system_prompt += "\nAge: " + age
    system_prompt += "\nSex: " + sex
    system_prompt += "\nPhysical Examination: " + pe
    system_prompt += "\nChoices: "+ choices
    return system_prompt

def get_onlyPE_mcq_prompt(age, sex, pe, choices):
    # Prompt used for Physical Exam + MCQ
    system_prompt = "You are an AI doctor specializing in dermatology."
    system_prompt += " You are given the patient's age, sex, physical examination results, and a list of possible diagnosis options."
    system_prompt += " Only one of the choices is correct."
    system_prompt += " Select the correct option, and give the answer as a short response. Do not explain."
    system_prompt += "\nAge: " + age
    system_prompt += "\nSex: " + sex
    system_prompt += "\nPhysical Examination: " + pe
    system_prompt += "\nChoices: " + choices
    return system_prompt

def get_frq_prompt(desc):
    # Prompt used for Vignette + FRQ and summarized conversation + FRQ
    system_prompt = "You are an AI doctor specializing in dermatology."
    system_prompt += " You are given the patient's symptoms."
    system_prompt += " Give the name of the correct diagnosis as a short answer. Do not explain."
    system_prompt += "\nSymptoms: " + desc
    return system_prompt

def get_frq_withPE_prompt(desc, age, sex, pe):
    # Prompt used for Vignette + FRQ and summarized conversation + FRQ
    system_prompt = "You are an AI doctor specializing in dermatology."
    system_prompt += " You are given the patient's symptoms."
    system_prompt += " Give the name of the correct diagnosis as a short answer. Do not explain."
    system_prompt += "\nSymptoms: " + desc
    system_prompt += "\nAge: " + age
    system_prompt += "\nSex: " + sex
    system_prompt += "\nPhysical Examination: " + pe
    return system_prompt

def get_onlyPE_frq_prompt(age, sex, pe):
    # Prompt used for Physical Exam + MCQ
    system_prompt = "You are an AI doctor specializing in dermatology."
    system_prompt += " You are given the patient's age, sex, and physical examination results."
    system_prompt += " Give the name of the correct diagnosis as a short answer. Do not explain."
    system_prompt += "\nAge: " + age
    system_prompt += "\nSex: " + sex
    system_prompt += "\nPhysical Examination: " + pe
    return system_prompt
