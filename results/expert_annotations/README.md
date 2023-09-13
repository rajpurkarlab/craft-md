### Expert Annotations

Dermatology experts (n=3) annotated 120 GPT-4 and GPT-3.5 multi-turn conversations to assess the performance of the doctor-AI, patient-AI and grader-AI agents. 

**File descriptions:**

`dermatologist_analysis_conversation_PE_frq_gpt3.tsv` and `dermatologist_analysis_conversation_PE_frq_gpt3.tsv`: Qualitative assessment of doctor-AI and patient-AI agents. Dermatologists were asked the following questions - 
q1: Can a definitive, unambiguous diagnosis be made solely through conversation?
q2: Does the conversation cover all essential details mentioned in the vignette, excluding those from the physical exam?
q3: Does the patient-AI agent communicate without using medical terminology terminology?


`dermatologist_annotations_case_vignette_qc.tsv` - A dermatology resident annotated each case vignette to determine whether there was a single definitive answer or one most likely answer among several possibilities. Annotations are available for each case vignette.


`dermatologist_annotations_publicdata_graderai_accuracy_assessment.tsv` - Dermatology experts (n=3) evaluated the diagnostic accuracy of the doctor-AI agent in experiments with FRQs, which included vignettes and multi-turn conversations, for both GPT-4 and GPT-3.5. Their results were then compared with the accuracy scores of the grader-AI to determine agreement between the dermatologist's evaluations and those of the grader-AI.

