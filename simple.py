# use supabase for storing chat_history online
from typing import Literal, List
import anthropic # type: ignore
import os
import yaml
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
        
        # Format history to maintain conversation flow in pairs
        formatted_history = []
        if chat_history:
            for msg in chat_history:
                if msg.get("user"):
                    formatted_history.append({
                        "role": "user",
                        "content": msg["user"]
                    })
                if msg.get("response"):
                    formatted_history.append({
                        "role": "assistant",
                        "content": msg["response"]
                    })
        
        # Add current query
        messages = formatted_history + [
            {"role": "user", "content": query}
        ]
        
        try:
            message = client.messages.create(
                system=prompts['MYCA'],
                model="claude-3-5-sonnet-20240620",
                max_tokens=400,
                temperature=0.9,
                messages=messages
            )
            return message.content[0].text
            
        except Exception as e:
            # Handle API-specific errors
            return "I apologize, but I'm having trouble processing your message. Could you please try again?"
            
    except Exception as e:
        # Handle general errors
        return "I encountered an error. Please try again or contact support if the issue persists."

# def chat(query: str, chat_history: List):
# # def chat(query: str, chat_history: list = [], name: str = 'Mukesh', gender: str = 'Male', occupation: str = 'Teacher'):     
#     # PROMPT = f"{prompts['MYCA']}\nPersonalise your responses based on the following user details. User name is {name}, user's gender is {gender} and occupation is {occupation}. Answer keeping these things in mind"
#     prompts = load_prompts()
    
#     PROMPT = f"{prompts['MYCA']} Here is the past user chat history: {chat_history}"
    
#     message = client.messages.create(
#             system=PROMPT,
#             model="claude-3-5-sonnet-20240620",
#             max_tokens=400,
#             temperature=0.9,
#             messages=[
#                 {
#                     "role": "user", 
#                     "content": query
#                 },
#                 # {
#                 #     "role": "assistant",
#                 #     "content": f"Here is the past chat history: {chat_history}" 
#                 # },
#             ],
#         )
#     ans =  message.content[0].text # message.content
#     return ans

# if __name__ == "__main__":
    
#     text = input("Please enter your topic: \n")
#     answer = chat(text)
    
#     # print(u"{}".format(answer))
#     print(u"Answer:\n{}".format(answer))

# Function to run a conversation turn
# def chat(message: str, config: dict, history: List):
#     try:
#         # Initialize messages list with optimal capacity
#         messages = []
#         # messages.reserve(len(history) * 2 + 1)  # Reserve space for history pairs + current message
#         # messages = [HumanMessage(content=msg["user"]) if msg.get("user") else None for msg in history] + [AIMessage(content=msg["response"]) if msg.get("response") else None for msg in history] + [HumanMessage(content=message)]
        
#         if history:
#             for msg in history:
#                 # Add messages in pairs to maintain conversation flow
#                 if msg.get("user"):
#                     messages.append(HumanMessage(content=msg["user"]))
#                 if msg.get("response"):
#                     messages.append(AIMessage(content=msg["response"]))
        
#         # Validate current message
#         if not message.strip():
#             raise ValueError("Empty message received")
        
#         # Add current message
#         messages.append(HumanMessage(content=message))
        
#         # Log for debugging (only in development)
#         logging.debug(f"Conversation context: {len(messages)} messages \n and FULL MESSAGE being {messages}")
        
#         # Invoke model with complete context
#         try:
#             result = graph.invoke({"messages": messages}, config=config)
#             return result["messages"][-1]
#         except Exception as e:
#             logging.error(f"Model invocation error: {str(e)}")
#             return AIMessage(content="I apologize, but I'm having trouble processing your message. Could you please try again?")
            
#     except Exception as e:
#         logging.error(f"Chat function error: {str(e)}")
#         return AIMessage(content="I encountered an error. Please try again or contact support if the issue persists.")
