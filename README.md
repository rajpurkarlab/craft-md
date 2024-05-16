# CRAFT-MD: Conversational Reasoning Assessment Framework for Testing in Medicine

This repository contains code and data for **C**onversational **R**easoning **A**ssessment **F**ramework for **T**esting in **M**e**d**icine (CRAFT-MD). CRAFT-MD is a novel framework designed to assess the conversational reasoning abilities of LLMs through simulated doctor-patient conversations that mimic clinical encounters. It evaluates clinical LLMs using a multi-agent framework comprising four components: a clinical LLM agent, a patient AI agent, a grader AI agent, and medical experts. The clinical LLM agent can be instantiated with any new LLM, allowing seamless evaluation of its clinical capabilities, such as history taking and information synthesis. The other agents play supportive roles in evaluating the clinical LLM agent.

[Paper Link:](https://www.medrxiv.org/content/10.1101/2023.09.12.23295399v2)

### Dataset

The dataset consists of 2000 questions, each structured as a case vignette followed by four answer choices (see Data Availability). Of these, 1800 were sourced from MedQA-USMLE, encompassing medical conditions commonly encountered in primary and specialist care settings. These questions span 12 medical specialties: Dermatology, Hematology and Oncology, Neurology, Gastroenterology, Pediatrics and Neonatology, Cardiology, Infectious Disease, Obstetrics and Gynecology, Urology and Nephrology, Endocrinology, Rheumatology, and Others. Additional 100 vignettes from an online question back and 100 newly generated vignettes are also included. The case vignettes were written by medical experts and contained details of the age and sex of the patient, current symptoms, medical history of illness and medications, and relevant family history. In some cases, physical exam and laboratory results were also included in the vignette.

### Replicating results

**Note: You will need to setup an OpenAI API setup, and replace the OpenAI key information in the below script.**
Jupyter notebooks can be run to replicate the CRAFT-MD framework and other results presented in the manuscript.


### System Requirements

All code contained in the repository was tested on python v3.9.17. The conda environment used can be re-created using `environment.yml`.  

All open-source models were tested on Quadro RTX 8000 48gb GPU. The GPT-4 and GPT-3.5 models can be tested on a normal computer.

## Citation

`Johri, S., Jeong, J., Tran, B. A., Schlessinger, D. I., Wongvibulsin, S., Cai, Z. R., ... & Rajpurkar, P. (2023). Guidelines For Rigorous Evaluation of Clinical LLMs For Conversational Reasoning. medRxiv, 2023-09.`

`Johri, Shreya, et al. "CRAFT-MD: A Conversational Evaluation Framework for Comprehensive Assessment of Clinical LLMs." AAAI 2024 Spring Symposium on Clinical Foundation Models. 2024.`

