import os
import json
from copy import deepcopy

from .utils import *
from .models import *
from .prompts import *

def craftmd_gpt(case, path_dir, doctor_model_name, num_runs = 5):

    case_id, case_desc, specialty, mcq_choices = case
    case_desc, question = get_case_without_question(case_desc)

    print(f'Thread for case id: {case_id} dispatched.')
    
    dmap_model = {"gpt4_1106": call_gpt4_api, "gpt3_1106": call_gpt3_api}
    doctor_model = dmap_model[doctor_model_name]
    patient_model = call_gpt4_api

    doctor_prompt = get_doctor_prompt(specialty)
    patient_prompt = get_patient_prompt(case_desc)

    stats = {}
    j = 0

    save_path = f'{path_dir}/{case_id}.json'
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            stats = json.load(f)
        while f'trial_{j}' in stats:
            j += 1
            
    if j == num_runs-1:
        return

    stats["case_vignette"] = case_desc
    stats["question"] = question
    
    while j < num_runs:
        stats[f"trial_{j}"] = {}
        flag = 0
        
        # Vignette + MCQ
        vignette_mcq_prompt = get_vignette_mcq_prompt(specialty, case_desc, question, mcq_choices)
        convo = [{"role": "system", "content": vignette_mcq_prompt}]
        vignette_mcq_ans = doctor_model(convo)
        stats[f"trial_{j}"]["vignette_mcq"] = vignette_mcq_ans
        
        # Vignette + FRQ
        vignette_frq_prompt = get_vignette_frq_prompt(specialty, case_desc, question)
        convo = [{"role": "system", "content": vignette_frq_prompt}]
        vignette_frq_ans = doctor_model(convo)
        stats[f"trial_{j}"]["vignette_frq"] = vignette_frq_ans
        
        # Multi-turn conversation experiments
        conversation_history_doctor = [{"role": "system", "content": doctor_prompt},
                                       {"role": "assistant", "content": "Hi! What symptoms are you facing today?"}]
        conversation_history_patient = [{"role": "system", "content": patient_prompt},
                                        {"role": "user", "content": "Hi! What symptoms are you facing today?"}]

        while True:
            # Patient talks
            response_patient = patient_model(conversation_history_patient) 
            if response_patient is None:
                flag=1
                break

            conversation_history_doctor.append({"role":"user",
                                               "content":response_patient})
            conversation_history_patient.append({"role":"assistant",
                                               "content":response_patient})

            # Doctor talks
            response_doctor = doctor_model(conversation_history_doctor)
            if response_doctor is None:
                flag=1
                break

            conversation_history_doctor.append({"role":"assistant",
                                               "content": response_doctor})
            conversation_history_patient.append({"role":"user", 
                                                "content": response_doctor})
            

            # Doctor arrives at a differential diagnosis
            if ("?" not in response_doctor) or ('Final Diagnosis' in response_doctor):
                stats[f"trial_{j}"]["multiturn_conversation"] = conversation_history_doctor
                break
        
        if flag == 0:
            # multi-turn conversation + MCQ
            prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
            multiturn_convo_without_diagnosis = conversation_history_doctor[:-1]
            multiturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            multiturn_mcq = doctor_model(multiturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["multiturn_mcq"] = multiturn_mcq

            # multi-turn conversation + FRQ
            prompt = get_frq_after_conversation_prompt(question)
            multiturn_convo_without_diagnosis = conversation_history_doctor[:-1]
            multiturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            multiturn_frq = doctor_model(multiturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["multiturn_frq"] = multiturn_frq

            # single-turn conversation + MCQ
            prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
            singleturn_convo_without_diagnosis = conversation_history_doctor[:3]
            singleturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            singleturn_mcq = doctor_model(singleturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["singleturn_mcq"] = singleturn_mcq

            # single-turn conversation + FRQ
            prompt = get_frq_after_conversation_prompt(question)
            singleturn_convo_without_diagnosis = conversation_history_doctor[:3]
            singleturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            singleturn_frq = doctor_model(singleturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["singleturn_frq"] = singleturn_frq
            
            # generate summarized conversation
            pat_responses = get_patient_responses(conversation_history_doctor[1:])
            summarized_conversation = convert_to_summarized(pat_responses)
            stats[f"trial_{j}"]["summarized_conversation"] = summarized_conversation

            # summarized conversation + MCQ
            summarized_mcq_prompt = get_vignette_mcq_prompt(specialty, summarized_conversation, question, mcq_choices)
            convo = [{"role": "user", "content": summarized_mcq_prompt}]
            summarized_mcq_ans = doctor_model(convo)
            stats[f"trial_{j}"]["summarized_mcq"] = summarized_mcq_ans

            # summarized conversation + FRQ
            summarized_frq_prompt = get_vignette_frq_prompt(specialty, summarized_conversation, question)
            convo = [{"role": "user", "content": summarized_frq_prompt}]
            summarized_frq_ans = doctor_model(convo)
            stats[f"trial_{j}"]["summarized_frq"] = summarized_frq_ans
            
            j += 1
        
        json.dump(stats, open(save_path, 'w'))
        
        
        
def craftmd_opensource(case, path_dir, doctor_model, doctor_tokenizer, num_runs = 5):

    case_id, case_desc, specialty, mcq_choices = case
    case_desc, question = get_case_without_question(case_desc)

    print(f'Thread for case id: {case_id} dispatched.')
    
    patient_model = call_gpt4_api

    doctor_prompt = get_doctor_prompt(specialty)
    patient_prompt = get_patient_prompt(case_desc)

    stats = {}
    j = 0

    save_path = f'{path_dir}/{case_id}.json'
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            stats = json.load(f)
        while f'trial_{j}' in stats:
            j += 1
            
    if j == num_runs-1:
        return

    stats["case_vignette"] = case_desc
    stats["question"] = question
    
    while j < num_runs:
        stats[f"trial_{j}"] = {}
        flag = 0
        
        # Vignette + MCQ
        vignette_mcq_prompt = get_vignette_mcq_prompt(specialty, case_desc, question, mcq_choices)
        conv = [{"role": "user", "content": vignette_mcq_prompt}]
        vignette_mcq_ans = call_open_llm(doctor_model, doctor_tokenizer, conv)
        stats[f"trial_{j}"]["vignette_mcq"] = vignette_mcq_ans

        # Vignette + FRQ
        vignette_frq_prompt = get_vignette_frq_prompt(specialty, case_desc, question)
        conv = [{"role": "user", "content": vignette_frq_prompt}]
        vignette_frq_ans = call_open_llm(doctor_model, doctor_tokenizer, conv)
        stats[f"trial_{j}"]["vignette_frq"] = vignette_frq_ans

        # Multi-turn conversation experiments
        conversation_history_doctor = [{"role": "user", "content": doctor_prompt},
                                       {"role": "assistant", "content": "Hi! What symptoms are you facing today?"}]
        conversation_history_patient = [{"role": "system", "content": patient_prompt},
                                        {"role": "user", "content": "Hi! What symptoms are you facing today?"}]

        while True:
            # Patient talks
            response_patient = patient_model(conversation_history_patient)
            if response_patient is None:
                flag=1
                break

            conversation_history_doctor.append({"role":"user",
                                               "content":response_patient})
            conversation_history_patient.append({"role":"assistant",
                                               "content":response_patient})


            # Doctor talks
            response_doctor = call_open_llm(doctor_model, doctor_tokenizer, conversation_history_doctor)
            if response_doctor is None:
                flag=1
                break

            conversation_history_doctor.append({"role":"assistant",
                                               "content": response_doctor})
            conversation_history_patient.append({"role":"user",
                                                "content": response_doctor})


            # Doctor arrives at a differential diagnosis
            if ("?" not in response_doctor) or ('Final Diagnosis' in response_doctor) or (len(conversation_history_doctor)>=30):
                stats[f"trial_{j}"]["multiturn_conversation"] = conversation_history_doctor
                break

        if flag==0:
            # multi-turn conversation + MCQ
            prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
            multiturn_convo_without_diagnosis = deepcopy(conversation_history_doctor[:-1])
            multiturn_convo_without_diagnosis[-1]["content"] += f"\n\n{prompt}"
            multiturn_mcq = call_open_llm(doctor_model, doctor_tokenizer, multiturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["multiturn_mcq"] = multiturn_mcq
    
            # multi-turn conversation + FRQ
            prompt = get_frq_after_conversation_prompt(question)
            multiturn_convo_without_diagnosis = deepcopy(conversation_history_doctor[:-1])
            multiturn_convo_without_diagnosis[-1]["content"] += f"\n\n{prompt}"
            multiturn_frq = call_open_llm(doctor_model, doctor_tokenizer, multiturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["multiturn_frq"] = multiturn_frq
    
            # single-turn conversation + MCQ
            prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
            singleturn_convo_without_diagnosis = deepcopy(conversation_history_doctor[:3])
            singleturn_convo_without_diagnosis[-1]["content"] += f"\n\n{prompt}"
            singleturn_mcq = call_open_llm(doctor_model, doctor_tokenizer, singleturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["singleturn_mcq"] = singleturn_mcq
    
            # single-turn conversation + FRQ
            prompt = get_frq_after_conversation_prompt(question)
            singleturn_convo_without_diagnosis = deepcopy(conversation_history_doctor[:3])
            singleturn_convo_without_diagnosis[-1]["content"] += f"\n\n{prompt}"
            singleturn_frq = call_open_llm(doctor_model, doctor_tokenizer, singleturn_convo_without_diagnosis)
            stats[f"trial_{j}"]["singleturn_frq"] = singleturn_frq
            
            # generate summarized conversation
            pat_responses = get_patient_responses(conversation_history_doctor[1:])
            summarized_conversation = convert_to_summarized(pat_responses)
            stats[f"trial_{j}"]["summarized_conversation"] = summarized_conversation

            # summarized conversation (with physical exam) + MCQ
            summarized_mcq_prompt = get_vignette_mcq_prompt(specialty, summarized_conversation, question, mcq_choices)
            conv = [{"role": "user", "content": summarized_mcq_prompt}]
            summarized_mcq_ans = call_open_llm(doctor_model, doctor_tokenizer, conv)
            stats[f"trial_{j}"]["summarized_mcq"] = summarized_mcq_ans

            # summarized conversation (with physical exam) + FRQ
            summarized_frq_prompt = get_vignette_frq_prompt(specialty, summarized_conversation, question)
            conv = [{"role": "user", "content": summarized_frq_prompt}]
            summarized_frq_ans = call_open_llm(doctor_model, doctor_tokenizer, conv)
            stats[f"trial_{j}"]["summarized_frq"] = summarized_frq_ans
    
            j += 1

        json.dump(stats, open(save_path, 'w'))
        
        
        
def craftmd_multimodal(case, img_dir, path_dir, deployment_name, num_runs = 5):

    case_id, case_desc, mcq_choices = case
    case_desc, question = get_case_without_question(case_desc)
    
    image_path = f'{img_dir}/{case_id.split("_")[1]}.jpeg'
    data_url = local_image_to_data_url(image_path)

    print(f'Thread for case id: {case_id} dispatched.')
    
    doctor_model = call_gpt4v_api
    patient_model = call_gpt4_api

    doctor_prompt = get_doctor_prompt_multimodal()
    patient_prompt = get_patient_prompt(case_desc)

    stats = {}
    j = 0

    save_path = f'{path_dir}/{case_id}.json'
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            stats = json.load(f)
        while f'trial_{j}' in stats:
            j += 1
            
    if j == num_runs-1:
        return

    stats["case_vignette"] = case_desc
    stats["question"] = question
    
    while j < num_runs:
        stats[f"trial_{j}"] = {}
        flag = 0
        
        # Vignette + MCQ
        vignette_mcq_prompt = get_vignette_mcq_prompt_multimodal(case_desc, question, mcq_choices)
        conv = [{"role": "system", "content": [{"type": "text",
                                                "text": vignette_mcq_prompt},
                                               {"type": "image_url",
                                                "image_url": {"url":data_url}}]}]
        vignette_mcq_ans = doctor_model(conv, deployment_name)
        
        # Vignette + FRQ
        vignette_frq_prompt = get_vignette_frq_prompt_multimodal(case_desc, question)
        conv = [{"role": "system", "content": [{"type": "text",
                                                "text": vignette_frq_prompt},
                                               {"type": "image_url",
                                                "image_url": {"url":data_url}}]}]
        vignette_frq_ans = doctor_model(conv, deployment_name)
        
        # Conversation experiments
        conversation_history_doctor = [{"role": "system", "content": [{"type": "text",
                                                                       "text": doctor_prompt},
                                                                      {"type": "image_url",
                                                                       "image_url": {"url": data_url}}]},
                                       {"role": "assistant", "content": "Hi! What symptoms are you facing today?"}]
        
        conversation_history_patient = [{"role": "system", "content": patient_prompt},
                                        {"role": "user", "content": "Hi! What symptoms are you facing today?"}]
        while True:
            # Patient talks
            response_patient = patient_model(conversation_history_patient) 
            if response_patient is None:
                flag=1
                break

            conversation_history_doctor.append({"role":"user",
                                               "content":response_patient})
            conversation_history_patient.append({"role":"assistant",
                                               "content":response_patient})

            # Doctor talks
            response_doctor = doctor_model(conversation_history_doctor, deployment_name)
            if response_doctor is None:
                flag=1
                break

            conversation_history_doctor.append({"role":"assistant",
                                               "content": response_doctor})
            conversation_history_patient.append({"role":"user", 
                                                "content": response_doctor})
            

            # Doctor arrives at a differential diagnosis
            if ("?" not in response_doctor) or ('Final Diagnosis' in response_doctor):
                stats[f"trial_{j}"]["multiturn_conversation"] = conversation_history_doctor
                break
        
        if flag == 0:
            # multi-turn conversation + MCQ
            prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
            multiturn_convo_without_diagnosis = conversation_history_doctor[:-1]
            multiturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            multiturn_mcq = doctor_model(multiturn_convo_without_diagnosis, deployment_name)
            stats[f"trial_{j}"]["multiturn_mcq"] = multiturn_mcq

            # multi-turn conversation + FRQ
            prompt = get_frq_after_conversation_prompt(question)
            multiturn_convo_without_diagnosis = conversation_history_doctor[:-1]
            multiturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            multiturn_frq = doctor_model(multiturn_convo_without_diagnosis, deployment_name)
            stats[f"trial_{j}"]["multiturn_frq"] = multiturn_frq

            # single-turn conversation + MCQ
            prompt = get_mcq_after_conversation_prompt(mcq_choices, question)
            singleturn_convo_without_diagnosis = conversation_history_doctor[:3]
            singleturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            singleturn_mcq = doctor_model(singleturn_convo_without_diagnosis, deployment_name)
            stats[f"trial_{j}"]["singleturn_mcq"] = singleturn_mcq

            # single-turn conversation + FRQ
            prompt = get_frq_after_conversation_prompt(question)
            singleturn_convo_without_diagnosis = conversation_history_doctor[:3]
            singleturn_convo_without_diagnosis.append({"role": "system", "content": prompt})
            singleturn_frq = doctor_model(singleturn_convo_without_diagnosis, deployment_name)
            stats[f"trial_{j}"]["singleturn_frq"] = singleturn_frq
            
            # generate summarized conversation
            pat_responses = get_patient_responses(conversation_history_doctor[1:])
            summarized_conversation = convert_to_summarized(pat_responses)
            stats[f"trial_{j}"]["summarized_conversation"] = summarized_conversation

            # summarized conversation + MCQ
            summarized_mcq_prompt = get_vignette_mcq_prompt(specialty, summarized_conversation, question, mcq_choices)
            convo = [{"role": "user", "content": summarized_mcq_prompt}]
            summarized_mcq_ans = doctor_model(convo, deployment_name)
            stats[f"trial_{j}"]["summarized_mcq"] = summarized_mcq_ans

            # summarized conversation + FRQ
            summarized_frq_prompt = get_vignette_frq_prompt(specialty, summarized_conversation, question)
            convo = [{"role": "user", "content": summarized_frq_prompt}]
            summarized_frq_ans = doctor_model(convo, deployment_name)
            stats[f"trial_{j}"]["summarized_frq"] = summarized_frq_ans
            
            j += 1
        
        json.dump(stats, open(save_path, 'w'))
        