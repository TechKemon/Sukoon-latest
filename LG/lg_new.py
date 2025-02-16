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
store = InMemoryStore()

# Define the state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define the conversational prompt
conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", prompts['MYCA']),
    ("system", "Context from past conversations:\n{memory_context}"),
    ("human", "{input}"),
])

class ConversationalAgent:
    def __init__(self, model, store: BaseStore, max_memories: int = 10):
        # self.prompt_template = prompt_template
        self.model = model
        self.store = store
        self.max_memories = max_memories
        # Cache the memory prompt template for reuse
        self.conversational_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts['MYCA']),
            ("system", "Context from past conversations:\n{memory_context}"),
            ("human", "{input}")
        ])
        
        # Initialize SQLite database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with required table"""
        conn = sqlite3.connect('db/conversations.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT,
                message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def fetch_conversations_from_sqlite(self, namespace):
        # Implement the logic to fetch past conversations from SQLite
        # For example, retrieve the last N messages for the user
        try:
            conn = sqlite3.connect('db/conversations.db')
            cursor = conn.cursor()
            cursor.execute("SELECT message FROM conversations WHERE namespace=? ORDER BY timestamp DESC LIMIT ?", (str(namespace),self.max_memories))
            rows = cursor.fetchall()
            conn.close()
            return [row[0] for row in rows]
        except Exception as e:
            logging.error(f"Error fetching conversations: {str(e)}")
            return []

    def summarize_conversations(self, conversations):
        try:
            if not conversations:
                return ""
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "Summarize the following conversations concisely for context:"),
                ("human", "\n".join(conversations)),
            ])
            summary = self.model.invoke(summary_prompt.format_messages())
            return summary.content.strip()
        except Exception as e:
            logging.error(f"Error summarizing conversations: {str(e)}")
            return ""

    # Fixed method to handle different message types
    def store_conversation_in_sqlite(self, namespace, messages):
        try:
            conn = sqlite3.connect('db/conversations.db')
            cursor = conn.cursor()
            for message in messages:
                # Handle different message types
                content = message.content if hasattr(message, 'content') else str(message)
                cursor.execute(
                    "INSERT INTO conversations (namespace, message, timestamp) VALUES (?, ?, ?)",
                    (str(namespace), content, datetime.now())
                )
            conn.commit()
            conn.close()
            
            # Cleanup old messages
            self._cleanup_old_messages(namespace)
        except Exception as e:
            logging.error(f"Error storing conversation: {str(e)}")

    # Added new method to cleanup old messages
    def _cleanup_old_messages(self, namespace):
        """Keep only the latest max_memories messages"""
        try:
            conn = sqlite3.connect('db/conversations.db')
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM conversations 
                WHERE namespace = ? 
                AND id NOT IN (
                    SELECT id FROM conversations 
                    WHERE namespace = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                )
            """, (str(namespace), str(namespace), self.max_memories))
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error cleaning up messages: {str(e)}")
    
# Updated main method to use class methods and handle streaming
    def run_conversational_agent(self, state: State):
        try:
            # Get user ID from state config
            user_id = state.get("configurable", {}).get("user_id", "default")
            namespace = f"memories_user_{user_id}"
            
            # Fetch and summarize past conversations
            past_conversations = self.fetch_conversations_from_sqlite(namespace)
            summary = self.summarize_conversations(past_conversations)
            
            # Format messages with context
            formatted_messages = self.conversational_prompt.format_messages(
                memory_context=summary,
                input=state["messages"][-1].content
            )
            
            # Get streaming response from model
            response = self.model.invoke(formatted_messages)
            
            # Store the conversation
            self.store_conversation_in_sqlite(namespace, state["messages"] + [response])
            
            return {"messages": state["messages"] + [response]}
            
        except Exception as e:
            logging.error(f"Error in conversational agent: {str(e)}")
            return {"messages": [AIMessage(content="I'm here to help. Could you please rephrase that?")]}

# Instantiate the conversational agent with tools
conversational_agent = ConversationalAgent(model, store, max_memories=10) # conversational_prompt, llm_with_tools
# agent = ConversationalAgent(prompt_template=conversational_prompt,model=model,store=store,max_memories=10)

# # Define the router function
# def route_query(state: State):
#     # Since we have only one agent now, we can directly route to the conversational agent
#     return "conversational"

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
graph = workflow.compile(checkpointer=memory, store=store)

# Function to run a conversation turn
def chat(message: str, config: dict):
    result = graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)
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
    return result["messages"][-1]

if __name__ == "__main__":
    # async def main():
    user_id = "1"
    config = {"configurable": {"thread_id": "1", "user_id": user_id}}
    chat_history = [] # KEPT EMPTY NOW
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("MYCA: Goodbye!")
            break
        
        response = chat(user_input, config, chat_history)
        chat_history.append({"user": user_input, "response": response.content})
        print("MYCA:") # end="\n"
        print(response.content)  # New line after complete response
        
        # print("MYCA:", end=" ", flush=True)
            # async for chunk in chat(user_input, config):
            #     # Handle different types of chunks
            #     if isinstance(chunk, str):
            #         print(chunk, end="", flush=True)
            #     elif isinstance(chunk, dict) and "content" in chunk:
            #         print(chunk["content"], end="", flush=True)
            #     elif hasattr(chunk, "content"):
            #         print(chunk.content, end="", flush=True)
            #     else:
            #         print(f"Debug: Received chunk type: {type(chunk)}")  # Debug line
    
    # async for event in app.astream_events({"messages": inputs}, version="v1"):
    # kind = event["event"]
    # print(f"{kind}: {event['name']}")
   
    # Run the async main function
    # asyncio.run(main())

# def run_conversational_agent(self, state: State):
    #     try:
    #         # Get user ID from config or state
    #         user_id = state.get("configurable", {}).get("user_id", "default")
    #         namespace = (user_id, "memories")
            
    #         # Fetch and manage memories efficiently
    #         memories = self._get_recent_memories(namespace)
    #         memory_context = self._format_memory_context(memories)
            
    #         # Format messages once
    #         formatted_messages = conversational_prompt.format_messages(
    #             memory_context=memory_context,
    #             input=state["messages"][-1].content
    #         )
            
    #         # Get response from model
    #         response = self.model.invoke(formatted_messages)
            
    #         # Asynchronously store the new memory
    #         self._store_memory(namespace, str(response))
            
    #         return {"messages": response}
            
    #     except Exception as e:
    #         # Log error and return graceful fallback response
    #         logging.error(f"Error in conversational agent: {str(e)}")
    #         return {"messages": self._get_fallback_response()}

    # def _get_recent_memories(self, namespace) -> list:
    #     """Efficiently retrieve recent memories with error handling"""
    #     try:
    #         memories = self.store.search(namespace)
    #         return memories[-self.max_memories:] if memories else []
    #     except Exception as e:
    #         logging.warning(f"Error fetching memories: {str(e)}")
    #         return []

    # def _format_memory_context(self, memories) -> str:
    #     """Format memories into context string with validation"""
    #     try:
    #         if not memories:
    #             return ""
    #         return "\n".join(
    #             m.value.get("user_message", "") 
    #             for m in memories 
    #             if isinstance(m.value, dict)
    #         )
    #     except Exception as e:
    #         logging.warning(f"Error formatting memories: {str(e)}")
    #         return ""
    
    # def _store_memory(self, namespace: tuple, memory: str) -> None:
    #     """Store memory with automatic cleanup of old memories"""
    #     try:
    #         # Store new memory
    #         memory_id = str(uuid.uuid4())
    #         self.store.put(namespace, memory_id, {
    #             "user_message": memory,
    #             "timestamp": datetime.now().isoformat()
    #         })
            
    #         # Cleanup old memories if needed
    #         self._cleanup_old_memories(namespace)
    #     except Exception as e:
    #         logging.error(f"Error storing memory: {str(e)}")

    # def _cleanup_old_memories(self, namespace: tuple) -> None:
    #     """Remove oldest memories when limit is exceeded"""
    #     try:
    #         memories = self.store.search(namespace)
    #         if len(memories) > self.max_memories:
    #             # Sort by timestamp and remove oldest
    #             sorted_memories = sorted(
    #                 memories,
    #                 key=lambda x: x.value.get("timestamp", ""),
    #                 reverse=True
    #             )
    #             for memory in sorted_memories[self.max_memories:]:
    #                 self.store.delete(namespace, memory.id)
    #     except Exception as e:
    #         logging.warning(f"Error cleaning up memories: {str(e)}")

    # def _get_fallback_response(self) -> AIMessage:
    #     """Return graceful fallback response if something goes wrong"""
    #     return AIMessage(content="I'm here to help. Could you please rephrase that?")