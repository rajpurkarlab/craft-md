import json
from .utils import get_correct_answer, diagnosis_evaluation

def graderai_evaluation(case_id, dataset, path_dir, experiment_names):
    print(f"Thread for {case_id} is dispatched.")
    
    save_path = f"{path_dir}/{case_id}.json"
    
    try:
        res = json.load(open(save_path,"r"))
    except Exception as e:
        print(e)
        return
    
    correct_ans = get_correct_answer(dataset, case_id)
    res["correct_ans"] = correct_ans
               
    for i in range(5):
        for exp in experiment_names:
#             check if the key exists already
            if f"evaluation_{exp}" in res[f"trial_{i}"].keys():
                continue 
            try:
                clinical_llm_response = res[f"trial_{i}"][exp]
            except:
                print(case_id, i, exp)
                res[f"trial_{i}"][f"extracted_ans_{exp}"] = "None"
                res[f"trial_{i}"][f"evaluation_{exp}"] = -1
                continue 
            
            if (clinical_llm_response is not None) and (type(clinical_llm_response) is not list): 
                res[f"trial_{i}"][f"evaluation_{exp}"], res[f"trial_{i}"][f"extracted_ans_{exp}"] = diagnosis_evaluation(correct_ans, clinical_llm_response)
            else:
                res[f"trial_{i}"][f"extracted_ans_{exp}"] = "None"
                res[f"trial_{i}"][f"evaluation_{exp}"] = -1  
                
    json.dump(res, open(save_path, 'w'))    
    
