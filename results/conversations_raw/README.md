### Raw multi-turn conversations

`conversations_gpt4.json` and `conversations_gpt3.json` contain the simulated multi-turn conversations generated using GPT-4 and GPT-3.5 respectively. Each file is formatted as follows - 

-`case_id`
    - `trial_{j}_doctor_responses_with_exam` - multi-turn conversations (with physical exam information provided at the end of the conversation)
    - `trial_{j}_doctor_responses` - multi-turn conversations (without physical exam)

j=0 to 9, representing the 10 trials per case_id.