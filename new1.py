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
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig

import os
import asyncio
import yaml, uuid
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY", default=None)
if not openai_api_key:
    load_dotenv(find_dotenv(), override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

def load_prompts(file_path='prompts.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

prompts = load_prompts()

# Initialize model with streaming enabled
model = ChatOpenAI(
    api_key=openai_api_key, 
    model="gpt-4o-mini",
    streaming=True # Enable streaming
)

# Initialize store
store = InMemoryStore()

# Define the state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class ConversationalAgent:
    def __init__(self, model, store: BaseStore, max_memories: int = 10):
        self.model = model
        self.store = store
        self.max_memories = max_memories
        self.conversational_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts['empathetic_agent_prompt']),
            ("system", "Context from past conversations:\n{memory_context}"),
            ("human", "{input}")
        ])

    # Now we can use a simpler async implementation
    async def run_conversational_agent(self, state: State):
        try:
            # Format messages
            formatted_messages = self.conversational_prompt.format_messages(
                memory_context="",  # Simplified for example
                input=state["messages"][-1].content
            )
            
            # Stream response directly
            response = await self.model.ainvoke(formatted_messages)
            return {"messages": response}
            
        except Exception as e:
            print(f"Error in conversational agent: {str(e)}")
            return {"messages": AIMessage(content="I'm here to help. Could you please rephrase that?")}

# Initialize agent
# conversational_agent = ConversationalAgent(model, store, max_memories=10)

# Create graph
# Create the graph with the async node
workflow = StateGraph(State)
workflow.add_node("conversational", ConversationalAgent(model, store).run_conversational_agent)
workflow.add_edge(START, "conversational")
workflow.add_edge("conversational", END)

# Compile graph
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory, store=store)

# Function to run conversation with streaming
# Simpler streaming chat function
async def chat(message: str, config: dict):
    print("MYCA:", end=" ", flush=True)
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=message)]},
        config=config,
        stream_mode="messages"
    ):
        if isinstance(chunk, AIMessageChunk):
            print(chunk.content, end="", flush=True)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("MYCA: Goodbye!")
            break
        
        asyncio.run(chat(user_input, config))
        print()