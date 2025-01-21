# use supabase for storing chat_history online

import anthropic # type: ignore
import os
import yaml
# set api_key
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

claude_api_key = os.getenv("ANTHROPIC_API_KEY", default=None)
if not claude_api_key:
    # Load just the .env file and try again
    load_dotenv(find_dotenv(), override=True)
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=claude_api_key)

# openai_api_key = os.getenv("OPENAI_API_KEY", default=None)
# if not openai_api_key:
#     load_dotenv(find_dotenv(), override=True)
#     openai_api_key = os.getenv("OPENAI_API_KEY")

def load_prompts(file_path='prompts.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def chat(query: str, chat_history: list = [], name: str = 'Mukesh', gender: str = 'Male', occupation: str = 'Teacher'):
    prompts = load_prompts()
    
    PROMPT = f"{prompts['MYCA']}\nPersonalise your responses based on the following user details. User name is {name}, user's gender is {gender} and occupation is {occupation}. Answer keeping these things in mind"
    
    message = client.messages.create(
            system=PROMPT,
            model="claude-3-5-sonnet-20240620",
            max_tokens=400,
            temperature=0.9,
            messages=[
                {
                    "role": "user", 
                    "content": query
                },
                {
                    "role": "assistant",
                    "content": f"Here is the past chat history: {chat_history}" 
                },
            ],
        )
    ans =  message.content[0].text # message.content
    return ans

# if __name__ == "__main__":
    
#     text = input("Please enter your topic: \n")
#     answer = chat(text)
    
#     # print(u"{}".format(answer))
#     print(u"Answer:\n{}".format(answer))
    