from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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
from langchain_anthropic import ChatAnthropic

import os
import yaml, uuid
import json
from datetime import datetime
import pandas as pd
import sqlite3
from typing import List, Dict
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

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

model = ChatOpenAI(api_key = openai_api_key)
# model = ChatAnthropic(api_key = anthropic_api_key)

in_memory_store = InMemoryStore()

# Initialize the model
# model = ChatOpenAI(
#     api_key=openai_api_key,
#     model="gpt-4o",
#     max_retries=2,
#     temperature=0.9,
#     max_tokens=150,
#     base_url=PORTKEY_GATEWAY_URL,
#     default_headers=createHeaders(
#         api_key=PORTKEY_API_KEY,
#         virtual_key=PORTKEY_VIRTUAL_KEY,
#         config = "pc-sukoon-86ab23"
#     )
# )

# Define the llama_index tool
# def llama_index(query: str):
#     """Use this tool to retrieve relevant information from the knowledge base."""
#     PERSIST_DIR = "./storage"
#     if not os.path.exists(PERSIST_DIR):
#         documents = SimpleDirectoryReader("data").load_data()
#         index = VectorStoreIndex.from_documents(documents)
#         index.storage_context.persist(persist_dir=PERSIST_DIR)
#     else:
#         storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
#         index = load_index_from_storage(storage_context)
    
#     query_engine = index.as_query_engine()
#     response = query_engine.query(query)
#     return {"messages": str(response)}

# tools = [llama_index]

# Bind tools to the model for the conversational agent
# llm_with_tools = model.bind_tools(tools)

# Define the state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define the conversational prompt
conversational_prompt = ChatPromptTemplate.from_messages([
    ("system", prompts['MYCA']),
    ("human", "{input}"),
])

# Create the ConversationalAgent class
class ConversationalAgent:
    def __init__(self, prompt_template, model):
        self.prompt_template = prompt_template
        self.model = model
        # self.store = store
        # self.namespace = ("memories", "1")
        # Clear memory when a new object is created
        # self.clear_old_memories()

    def run_conversational_agent(self, state: State, store: BaseStore):
        namespace = (user_id, "memories")
        memory_id = str(uuid.uuid4())
        memories = store.search(namespace)
        info = "\n".join([d.value["user_message"] for d in memories])
        
        # Create a new prompt template with memory context
        prompt_with_memory = ChatPromptTemplate.from_messages([
            ("system", prompts['MYCA']),
            ("system", "Context from past conversations:\n{memory_context}"),
            ("human", "{input}")
        ])
        
        # Get the last human message from state
        last_message = state["messages"][-1].content
        
        # Format the prompt with memory context
        formatted_messages = prompt_with_memory.format_messages(
            memory_context=info,
            input=last_message
        )
        
        # Get response from the model
        response = self.model.invoke(formatted_messages)
        
        # Retrieve the last 10 memories if available
        # last_memories = memories[-10:] if len(memories) > 10 else memories
        # system_msg = f"{conversational_prompt} \n\n ### Take into account these past conversations when responding as MYCA: {info}"
        # response = model.invoke(
        #     [{"type": "system", "content": system_msg}] + state["messages"]
        # )
        
        # Store new memories if the user asks the model to remember
        # last_message = state["messages"][-1]
        # if "remember" in last_message.content.lower():
        # memory = {"user_msg" : {response}}
        memory = str(response)
        in_memory_store.put(namespace, memory_id, {"user_message": memory})
        # self.store.put(self.namespace, str(uuid.uuid4()), {"user_message": memory})
        # self.clean_old_memories()
        return {"messages": response}
        # return {"messages": [f"LLM received {state['messages']=}"]}
   
#    conversational_agent = ConversationalAgent(conversational_prompt, model, BaseStore)
    
    # def run_conversational_agent(self, state: State):
    #     # print("Running conversational agent")
    #     convo_model = conversational_prompt | model # model
    #     response = convo_model.invoke(state["messages"])
    #     return {"messages": response}
    
    # def run(self, state: State):
    #     print("Running ConversationalAgent")
    #     messages = state["messages"]
    #     last_message = messages[-1]
    #     formatted_messages = self.prompt_template.format_messages(input=last_message.content)
    #     # Use the combined model with tools
    #     response = self.model.invoke(formatted_messages)
    #     # Append the response to the state messages
    #     state["messages"].append(AIMessage(content=response.content))
    #     return {"messages": state["messages"]}
    
    # NOTE: SUMMARIZE PAST CONVOS USING SQLITE3 & AI
    # def run_conversational_agent(state: State):
    #     # Fetch past conversations from SQLite
    #     namespace = ("memories_user_id", "1")
    #     past_conversations = fetch_conversations_from_sqlite(namespace)

    #     # Summarize past conversations to keep input concise
    #     summary = summarize_conversations(past_conversations, model)

    #     # Include the summary in the system message
    #     system_msg = f"{conversational_prompt}\n\nSummary of previous conversations: {summary}"

    #     # Format messages for the model
    #     messages = [SystemMessage(content=system_msg)] + state["messages"]

    #     # Invoke the model
    #     response = model.invoke(messages)

    #     # Optionally, store the current conversation to SQLite
    #     store_conversation_in_sqlite(namespace, state["messages"] + [response])

    #     return {"messages": state["messages"] + [response]}

    # def fetch_conversations_from_sqlite(namespace):
    #     # Implement the logic to fetch past conversations from SQLite
    #     # For example, retrieve the last N messages for the user
    #     conn = sqlite3.connect('db/conversations.db')
    #     cursor = conn.cursor()
    #     LIMIT = 10
    #     cursor.execute("SELECT message FROM conversations WHERE namespace=? ORDER BY timestamp DESC LIMIT ?", (namespace,LIMIT))
    #     rows = cursor.fetchall()
    #     conn.close()
    #     return [row[0] for row in rows]

    # def summarize_conversations(conversations, model):
    #     # Use the model to summarize past conversations
    #     summary_prompt = ChatPromptTemplate.from_messages([
    #         ("system", "Summarize the following conversations concisely for context:"),
    #         ("human", "\n".join(conversations)),
    #     ])
    #     summary = model.invoke(summary_prompt.format_messages())
    #     return summary.content.strip()

    # def store_conversation_in_sqlite(namespace, messages):
    #     # Implement logic to store the conversation in SQLite
    #     conn = sqlite3.connect('db/conversations.db')
    #     cursor = conn.cursor()
    #     for message in messages:
    #         cursor.execute("INSERT INTO conversations (namespace, message, timestamp) VALUES (?, ?, ?)",
    #                        (namespace, message.content, datetime.now()))
    #     conn.commit()
    #     conn.close()

# Instantiate the conversational agent with tools
conversational_agent = ConversationalAgent(conversational_prompt, model) # conversational_prompt, llm_with_tools

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
graph = workflow.compile(checkpointer=memory, store=in_memory_store)

# Function to run a conversation turn
def chat(message: str, config: dict):
    result = graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    return result["messages"][-1]

if __name__ == "__main__":
    # config = {"configurable": {"thread_id": "1"}}
    user_id = "1"
    config = {"configurable": {"thread_id": "1", "user_id": user_id}}
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Sukoon: Goodbye!")
            break
        response = chat(user_input, config)
        print("Sukoon:", response.content)
        
        # for update in graph.stream(
        #     {"messages": [{"role": "user", "content": f"{user_input}"}]}, config, stream_mode="updates"
        # ):
        #     print(update)