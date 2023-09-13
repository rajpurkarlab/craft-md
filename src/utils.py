import openai
import time

def call_gpt3_api(convo, n_responses, max_tokens=100):
    try:
        response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-0301",
                        messages = convo,
                        max_tokens = max_tokens,
                        n = n_responses
                        )
        if n_responses == 1:
            responses = response["choices"][0]["message"]['content']
        else:
            responses = [response["choices"][i]["message"]['content'] for i in range(n_responses)]
            
        return responses

    except Exception as e:
        print(e)
        return call_gpt3_api(convo, n_responses, max_tokens)
               
    

def call_gpt4_api(convo, n_responses, max_tokens=100):
    try:
        response = openai.ChatCompletion.create(
                        model="gpt-4-0314",
                        messages = convo,
                        max_tokens = max_tokens,
                        n = n_responses
                        )
        if n_responses==1:
            responses = response["choices"][0]["message"]['content']
        else:
            responses = [response["choices"][i]["message"]['content'] for i in range(n_responses)]
            
        return responses

    except Exception as e:
        print(e)
        return call_gpt4_api(convo, n_responses, max_tokens)
