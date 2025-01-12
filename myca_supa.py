from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AIMessageChunk
from pydantic import BaseModel, Field
from langgraph.graph.message import AnyMessage, add_messages
from typing import Literal, Annotated
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from portkey_ai.langchain import LangchainCallbackHandler
# from langchain_anthropic import ChatAnthropic

import os
import asyncio
import yaml, uuid
import json
from datetime import datetime
import pandas as pd
import sqlite3
from typing import List, Dict
from pathlib import Path
import logging

from dotenv import load_dotenv, find_dotenv

openai_api_key = os.getenv("OPENAI_API_KEY", default=None)
if not openai_api_key:
    # Load just the .env file and try again
    load_dotenv(find_dotenv(), override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    # PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    # PORTKEY_VIRTUAL_KEY = os.getenv("PORTKEY_VIRTUAL_KEY")
    # PORTKEY_VIRTUAL_KEY_A = os.getenv("PORTKEY_VIRTUAL_KEY_A")

def load_prompts(file_path='prompts.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
prompts = load_prompts()

model = ChatOpenAI(api_key = openai_api_key, model="gpt-4o-mini", temperature=0.7, max_tokens=300, max_retries=2) # , streaming=True
# model = ChatAnthropic(model="claude-3-5-haiku-20241022", api_key=anthropic_api_key, max_tokens=300, max_retries=2, temperature=0.7)

# in_memory_store = InMemoryStore()
# store = InMemoryStore()

# Define the state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define the conversational prompt
conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", prompts['MYCA']),
    # ("system", "Context from past conversations:\n{memory_context}"),
    ("human", "{input}"),
])

class ConversationalAgent:
    def __init__(self, model):
        self.model = model
        # Single prompt template with efficient history handling
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompts['MYCA']),
            ("human", "{input}")
        ])

    def run_conversational_agent(self, state: State):
        try:
            # The messages list already contains the full conversation history
            # No need to reformat it since it's already in the correct LangChain format
            messages = state["messages"]
            
            # Create system message with the existing context
            system_message = SystemMessage(content=prompts['MYCA'])
            
            # Combine system message with existing conversation history
            formatted_messages = [system_message] + messages
            
            # Get response from model
            response = self.model.invoke(formatted_messages)
            return {"messages": state["messages"] + [response]}
            
        except Exception as e:
            logging.error(f"Error in conversational agent: {str(e)}")
            return {"messages": [AIMessage(content="I'm here to help. Could you please rephrase that?")]}

# class ConversationalAgent:
#     def __init__(self, model): # , store: BaseStore, max_memories: int = 10
#         # self.prompt_template = prompt_template
#         self.model = model
#         # self.store = store
#         # self.max_memories = max_memories
        
#         self.system_history_prompt = ChatPromptTemplate.from_messages([
#             ("system", prompts['MYCA'] + "\n\nPrevious conversation history:\n{history}"),
#             ("human", "{input}")
#         ])
        
#         # self.user_history_prompt = ChatPromptTemplate.from_messages([
#         #     ("system", prompts['MYCA']),
#         #     ("human", "Previous conversation:\n{history}\n\nCurrent message: {input}")
#         # ])
#         # Cache the memory prompt template for reuse
#         self.conversational_prompt = ChatPromptTemplate.from_messages([
#             ("system", prompts['MYCA']),
#             # ("system", "Context from past conversations:\n{memory_context}"), # change it to HISTORY
#             ("human", "{input}")
#         ])

#     def format_history(self, messages):
#         """Format the message history into a readable string"""
#         formatted_history = []
#         for msg in messages[:-1]:  # Exclude the last message as it's the current input
#             if isinstance(msg, HumanMessage):
#                 formatted_history.append(f"Human: {msg.content}") # is this right approach here?
#             elif isinstance(msg, AIMessage):
#                 formatted_history.append(f"Assistant: {msg.content}")
#         return "\n".join(formatted_history)
    
#     def summarize_conversations(self, conversations):
#         try:
#             if not conversations:
#                 return ""
#             summary_prompt = ChatPromptTemplate.from_messages([
#                 ("system", "Summarize the following conversations concisely for context:"),
#                 ("human", "\n".join(conversations)),
#             ])
#             summary = self.model.invoke(summary_prompt.format_messages())
#             return summary.content.strip()
#         except Exception as e:
#             logging.error(f"Error summarizing conversations: {str(e)}")
#             return ""
        
# # Updated main method to use class methods and handle streaming
#     def run_conversational_agent(self, state: State):
#         try:
#             # Get user ID from state config
#             # user_id = state.get("configurable", {}).get("user_id", "default")
#             # namespace = f"memories_user_{user_id}"
            
#             # Format the conversation history
#             history = self.format_history(state["messages"])
#             logging.info(f"MESSAGE IN CONVO AGENT IS: {history}")
#             current_input = state["messages"][-1].content
            
#             # # Fetch and summarize past conversations
#             # past_conversations = self.fetch_conversations_from_sqlite(namespace)
#             # summary = self.summarize_conversations(past_conversations)
            
#             formatted_messages = self.system_history_prompt.format_messages( # use user_history_prompt if want to add history to prompt
#                 history=history,
#                 input=current_input
#             )
            
#             # Format messages with context
#             # formatted_messages = self.conversational_prompt.format_messages(
#             #     memory_context=summary,
#             #     input=state["messages"][-1].content
#             # )
            
#             # Get streaming response from model
#             # response = self.model.invoke(formatted_messages)
#             # response = self.model.invoke(state["messages"][-1].content)
            
#             # Store the conversation
#             # self.store_conversation_in_sqlite(namespace, state["messages"] + [response])
            
#             # return {"messages": state["messages"] + [response]}
            
#             response = self.model.invoke(formatted_messages)
#             return {"messages": state["messages"] + [response]}
            
#         except Exception as e:
#             logging.error(f"Error in conversational agent: {str(e)}")
#             return {"messages": [AIMessage(content="I'm here to help. Could you please rephrase that?")]}

# Instantiate the conversational agent with tools
# conversational_agent = ConversationalAgent(model, store, max_memories=10) # conversational_prompt, llm_with_tools
conversational_agent = ConversationalAgent(model)
# agent = ConversationalAgent(prompt_template=conversational_prompt,model=model,store=store,max_memories=10)

# Create the graph
workflow = StateGraph(State)
workflow.add_node("conversational", conversational_agent.run_conversational_agent)
# workflow.add_conditional_edges(
#     START,
#     route_query,
#     {"conversational": "conversational"}
# )
workflow.add_edge(START,"conversational")
workflow.add_edge("conversational", END)

# Compile the graph
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
# graph = workflow.compile(checkpointer=memory, store=store)

# Function to run a conversation turn
def chat(message: str, config: dict, history: List):
    try:
        # Initialize messages list with optimal capacity
        messages = []
        # messages.reserve(len(history) * 2 + 1)  # Reserve space for history pairs + current message
        # messages = [HumanMessage(content=msg["user"]) if msg.get("user") else None for msg in history] + [AIMessage(content=msg["response"]) if msg.get("response") else None for msg in history] + [HumanMessage(content=message)]
        
        if history:
            for msg in history:
                # Add messages in pairs to maintain conversation flow
                if msg.get("user"):
                    messages.append(HumanMessage(content=msg["user"]))
                if msg.get("response"):
                    messages.append(AIMessage(content=msg["response"]))
        
        # Validate current message
        if not message.strip():
            raise ValueError("Empty message received")
        
        # Add current message
        messages.append(HumanMessage(content=message))
        
        # Log for debugging (only in development)
        logging.debug(f"Conversation context: {len(messages)} messages \n and FULL MESSAGE being {messages}")
        
        # Invoke model with complete context
        try:
            result = graph.invoke({"messages": messages}, config=config)
            return result["messages"][-1]
        except Exception as e:
            logging.error(f"Model invocation error: {str(e)}")
            return AIMessage(content="I apologize, but I'm having trouble processing your message. Could you please try again?")
            
    except Exception as e:
        logging.error(f"Chat function error: {str(e)}")
        return AIMessage(content="I encountered an error. Please try again or contact support if the issue persists.")

# def chat(message: str, config: dict, history: List):
#     try:
#     # Convert history messages to LangChain message format
#         messages = []
#         if history:
#                 for msg in history:
#                     # Add user message if it exists and is not empty
#                     if msg.get("user"):
#                         logging.info(f"Adding user message to history: {msg['user']}")
#                         messages.append(HumanMessage(content=msg["user"]))
#                     # Add AI response if it exists and is not empty
#                     if msg.get("response"):
#                         logging.info(f"Adding AI response to history: {msg['response']}")
#                         messages.append(AIMessage(content=msg["response"]))
#         else:
#             logging.info(f"Unable to add this convo as context {history}")
#             print("\nFAILURE\n")
#         # Add current message
#         if not message.strip():
#             raise ValueError("Empty message received")
        
#         # Add current message
#         messages.append(HumanMessage(content=message))
#         logging.info(f"\n current FULL MESSAGE is: \n{messages}\n")
#         # Invoke the model with messages and config
#         try:
#             result = graph.invoke({"messages": messages}, config=config)
#             return result["messages"][-1]
#         except Exception as e:
#             logging.error(f"Error invoking model: {str(e)}")
#             return AIMessage(content="I apologize, but I'm having trouble processing your message. Could you please try again?")
        
#     except Exception as e:
#         logging.error(f"Error in chat function: {str(e)}")
#         return AIMessage(content="I encountered an error. Please try again or contact support if the issue persists.")
    # for update in graph.stream(
    #         {"messages": [HumanMessage(content=message)]}, config=config, stream_mode="messages" # updates, values, debug
    #     ):
    #     # print(update, "\n", type(update))
    #     # print()
    #     if "messages" in update and update["messages"]:
    #         # Yield each chunk of the response
    #         ai_message = update['conversational']['messages']
    #         # if isinstance(ai_message, AIMessageChunk):
    #         print(ai_message.content, end="", flush=True)
                # yield ai_message
    # return result["messages"][-1]

# if __name__ == "__main__":
#     # async def main():
#     user_id = "1"
#     config = {"configurable": {"thread_id": "1", "user_id": user_id}}
#     chat_history = [] # KEPT EMPTY NOW
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("MYCA: Goodbye!")
#             break
        
#         response = chat(user_input, config, chat_history)
#         chat_history.append({"user": user_input, "response": response.content})
#         print("MYCA:") # end="\n"
#         print(response.content)  # New line after complete response