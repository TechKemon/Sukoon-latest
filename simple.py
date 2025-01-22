# use supabase for storing chat_history online
from typing import Literal, List
import anthropic # type: ignore
import os, json
import yaml
# from pydantic import BaseModel, Field
# from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
# set api_key
from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

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

def chat(query: str, chat_history: List) -> str:
    try:
        # Validate current message
        if not query.strip():
            raise ValueError("Empty message received")

        prompts = load_prompts()
        prompt = f"{prompts['MYCA']}"
        
        # print(f"Received chat_history: {json.dumps(chat_history, indent=2)}")
        # Format history to maintain conversation flow in pairs
        history = []
        if chat_history:
            for msg in chat_history:
                if msg.get("user"):
                    history.append({
                        "role": "user",
                        "content": msg["user"]
                    })
                if msg.get("response"):
                    history.append({
                        "role": "assistant",
                        "content": msg["response"]
                    })
        
        # Add current query to history
        history.append({"role": "user", "content": query})
        # print(f"FULL CONVERSATION HISTORY is \n {history} \n")
        
        # # Add current query
        # system_prompt = f"{prompts['MYCA']}\nPrevious conversation history:\n{formatted_history}"
        
        # chat = [
        #     {"role": "user", "content": query}
        # ]
        
        try:
            response = client.messages.create(
                system=[{
                    "type": "text",
                    "text": prompts['MYCA'],
                    "cache_control": {"type": "ephemeral"}
                }], # prompt
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                temperature=0.9,
                messages=history
            )
            answer = response.content[0].text
            # print(answer) # for debugging
            return answer
            
        except Exception as e:
            # Handle API-specific errors
            return "I apologize, but I'm having trouble processing your message. Could you please try again?"
            
    except Exception as e:
        # Handle general errors
        return "I encountered an error. Please try again or contact support if the issue persists."