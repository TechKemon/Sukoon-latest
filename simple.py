# use supabase for storing chat_history online
from typing import Literal, List
import anthropic # type: ignore
from openai import OpenAI
# import google.generativeai as genai
import os
import yaml
# from pydantic import BaseModel, Field
# from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
# set api_key
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# using a constant for model names
# MODEL_NAMES = {
#     "OPENAI": "gpt-4o",
#     "CLAUDE": "claude-3-5-sonnet-20241022",
#     "GEMINI": "gemini-1.5-flash"
# }

claude_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Reload .env if any API key is missing
if not (claude_api_key and openai_api_key):
    load_dotenv(find_dotenv(), override=True)
    claude_api_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

client_claude = anthropic.Anthropic(api_key=claude_api_key) if claude_api_key else None
client_openai = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Initialize Gemini client
# gemini_api_key = os.getenv("GEMINI_API_KEY")
# if gemini_api_key:
#     genai.configure(api_key=gemini_api_key)
# else:
#     genai = None  # or handle absence

def format_history(chat_history: List) -> List:
    """Formats chat history into API-compatible message format."""
    # Format history to maintain conversation flow in pairs
        # history = []
        # if chat_history:
        #     for msg in chat_history:
        #         if msg.get("user"):
        #             history.append({
        #                 "role": "user",
        #                 "content": msg["user"]
        #             })
        #         if msg.get("response"):
        #             history.append({
        #                 "role": "assistant",
        #                 "content": msg["response"]
        #             })
    formatted = []
    for msg in chat_history:
        if msg.get("user"):
            formatted.append({"role": "user", "content": msg["user"]})
        if msg.get("response"):
            formatted.append({"role": "assistant", "content": msg["response"]})
    return formatted

def load_prompts(file_path='prompts.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def chat(query: str, chat_history: List) -> str:
    try:
        # Validate current message
        if not query.strip():
            return "Please enter a valid message."

        prompts = load_prompts()
        # prompt = f"{prompts['MYCA']}"
        system_prompt = prompts['MYCA'] 
        history = format_history(chat_history)
        # Add current query to history
        history.append({"role": "user", "content": query})
        # print(f"Received chat_history: {json.dumps(chat_history, indent=2)}")
        # print(f"FULL CONVERSATION HISTORY is \n {history} \n")
        # # Add current query
        # system_prompt = f"{prompts['MYCA']}\nPrevious conversation history:\n{formatted_history}"
        
        # chat = [
        #     {"role": "user", "content": query}
        # ]
        
        try:
            if client_openai:
                response = client_openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": system_prompt}] + history,
                    temperature=0.9,
                    max_tokens=1024
                )
                answer = response.choices[0].message.content
                print("answer from openai is :", answer)
                return answer
            else:
                raise RuntimeError("OpenAI client not initialized")
        
        except Exception as e:
            print(f"OpenAI Error: {str(e)}. Falling back to Claude...")
            
            try:
                if client_claude:
                    response = client_claude.messages.create(
                        system=[{
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"}
                        }], # prompt
                        model="claude-3-5-haiku-latest", #  claude-3-5-sonnet-20241022
                        max_tokens=1024,
                        temperature=0.9,
                        messages=history
                    )
                    answer = response.content[0].text
                    # print(answer) # for debugging
                    return answer  
            except Exception as e:
                print(f"Claude Error: {str(e)}. Falling back to Gemini...")
                # gemini method
                # try:
                #     model = clients["gemini"].GenerativeModel('gemini-pro')
                #     chat = model.start_chat()
                #     full_prompt = f"{system_prompt}\n\nConversation History:\n" + "\n".join(
                #         [f"{msg['role']}: {msg['content']}" for msg in history]
                #     )
                #     response = chat.send_message(full_prompt)
                #     return response.text
                # except Exception as e:
                #     print(f"Gemini API Error: {e}")
                return "I apologize, but I'm having trouble processing your message. Could you please try again?"
        
    except Exception as e:
        # Handle general errors
        return "I encountered an error. Please try again or contact support if the issue persists."