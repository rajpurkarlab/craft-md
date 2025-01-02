# CRAFT-MD

## An Evaluation Framework for Conversational Reasoning in Clinical LLMs During Patient Interactions

<!-- [Nature Medicine Paper](link) | [Live Benchmark](link) | [AAAI '24](https://openreview.net/forum?id=Bk2nbTDtm8)| [Cite Us](https://github.com/rajpurkarlab/craft-md?tab=readme-ov-file#citation) -->

<details>
  <summary>
	  <b>Main Findings</b>
  </summary>
CRAFT-MD is a robust and scalable evaluation framework designed to assess the conversational reasoning capabilities of clinical Large Language Models (LLMs) in real-world scenarios, going beyond traditional accuracy metrics derived from exam-style questions. The framework simulates doctor-patient interactions, where the clinical LLM's ability to gather medical histories, synthesize information, and arrive at accurate diagnoses is evaluated through a multi-agent setup. This setup includes a patient-AI, a grader-AI, and validation by medical experts to ensure the reliability of the results.
	
<p></p>Based on the evaluation of leading commercial and open-source LLMs, we propose the following recommendations to enhance the assessment of their clinical capabilities -
<p></p>

| Recommendation | Description |
|----------------|-------------|
| Recommendation 1 | Evaluate diagnostic accuracy through realistic doctor-patient conversations. |
| Recommendation 2 | Employ open-ended questions for evaluating diagnostic reasoning. |
| Recommendation 3 | Assess comprehensive history taking skills. |
| Recommendation 4 | Evaluate LLMs on the synthesis of information over multiple dialogues. |
| Recommendation 5 | Incorporate multimodal information available to physicians to enhance LLM performance. |
| Recommendation 6 | Continuous evaluation of conversational abilities for guiding development of clinical LLMs. |
| Recommendation 7 | Test and refine prompting strategies to enhance LLM performance. |
| Recommendation 8 | Implement patient-LLM interactions for ethical and scalable testing. |
| Recommendation 9 | Combine automated and expert evaluations for comprehensive insights. |
| Recommendation 10 | Encourage collection of public datasets covering diverse medical scenarios, suited for open-ended evaluation. |

</details>

### Updates
<!-- - Jan 2025 : Our Nature Medicine paper is online ✨. We are also releasing a [Live Benchmark]() with new models! Track the progress of clinical reasoning capabilties of LLMs with us. -->
- May 2024 : We got selected for a Poster Presentation at _SAIL 2024_.
- Mar 2024 : We got selected for an Oral Presentation at _AAAI Spring Symposium for Clinical Foundation Models_ and won the **Best Paper Award**!🏆
- Jan 2024 : We've updated our preprint with more models.
- Aug 2023 : Our preprint is available online!

### Reproducing Results
**Note:** You will need to configure the OpenAI API and replace the placeholder OpenAI key in the provided scripts.

* `data/` - Contains the MedQA-USMLE, Derm-Public, and Derm-Private datasets. URLs to all NEJM Image Challenge cases used in this study are also provided.
* `results/` - Stores the results for each evaluated model within their respective subfolders. Each case's results are saved as `.json` files for all five trials, with a key for each CRAFT-MD experiment.
* `src/` - Includes the codebase used for evaluations conducted with CRAFT-MD.

To facilitate replication of our study's results, we provide the following scripts:
* `parallel_craftmd_gpt.py` - For replicating results of GPT-4 and GPT-3.5.
* `parallel_craftmd_multimodal.py` - For replicating results of GPT-4V.
* `craftmd_opensource.py` - For replicating results of LLaMA-2-7b, Mistral-v1, and Mistral-v2.

All code was tested with Python v3.9.17. You can recreate the environment using the provided `environment.yml` file. All open-source models were tested on Quadro RTX 8000 48gb GPU. The GPT-4 and GPT-3.5 models can be run on a personal laptops as well.

<!-- ### Citation
If you've found this work useful, please cite the following :

Johri, S., Jeong, J., Tran, B. A., Schlessinger, D. I., Wongvibulsin, S., Barnes, L. A., ... & Rajpurkar, P. (2023). An Evaluation Framework for Conversational Reasoning in Clinical LLMs During Patient Interactions. Nat Med (2025). 
```
@article{johri2025craftmd,
  title={An Evaluation Framework for Conversational Reasoning in Clinical LLMs During Patient Interactions},
  author={Johri, Shreya and Jeong, Jaehwan and Tran, Benjamin A. and Schlessinger, Daniel I. and Wongvibulsin, Shannon and Barnes, Leandra A. and Zhou, Hong-Yu and Cai, Zhou Ran and others},
  journal={Nature Medicine},
  publisher={Nature Publishing Group},
  year={2025}
}
```
 -->
### Issues
Please report issues directly to sjohri@g.harvard.edu.

