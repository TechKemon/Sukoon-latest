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
# from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from portkey_ai.langchain import LangchainCallbackHandler
from langchain_anthropic import ChatAnthropic

import redis
import os
import yaml, uuid
import json
from datetime import datetime
import pandas as pd
import sqlite3
from typing import List, Dict
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
PORTKEY_VIRTUAL_KEY = os.getenv("PORTKEY_VIRTUAL_KEY")
PORTKEY_VIRTUAL_KEY_A = os.getenv("PORTKEY_VIRTUAL_KEY_A")

def load_prompts(file_path='prompts.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

prompts = load_prompts()

# define memory object
# in_memory_store = InMemoryStore()
# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# PORTKEY IMPLEMENTATION
portkey_handler = LangchainCallbackHandler(
    api_key=PORTKEY_API_KEY,
    metadata={
        "session_id": "session_1",  # Use consistent metadata across your application
        "agent_id": "Router_Agent",  # Specific to the current agent
    }
)
# Initialize OpenAI model
# model = llm
# model = ChatOpenAI(model="gpt-4o", max_tokens=50, temperature=0.9, max_retries=2)
model = ChatOpenAI(
    api_key=openai_api_key,
    model="gpt-4o",
    max_retries=2,
    temperature=0.9,
    max_tokens=150,
    base_url=PORTKEY_GATEWAY_URL,
    default_headers=createHeaders(
        api_key=PORTKEY_API_KEY,
        virtual_key=PORTKEY_VIRTUAL_KEY, # Pass your virtual key saved on Portkey for any provider you'd like (Anthropic, OpenAI, Groq, etc.). if using this, no need to pass openai api key
        config = "pc-sukoon-86ab23"
    ),
    callbacks=[portkey_handler],
)
# max_completion_tokens = 200 for o1 models

# model_a = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0.9, api_key=anthropic_api_key, max_tokens = 200, max_retries=2) 
# to use sonnet 3.5, put claude-3-5-sonnet-20241022 as model
model_a = ChatOpenAI(
    api_key=anthropic_api_key, # We'll pass a dummy API key here
    base_url=PORTKEY_GATEWAY_URL,
    default_headers=createHeaders(
        api_key=PORTKEY_API_KEY,
        virtual_key=PORTKEY_VIRTUAL_KEY_A, # Pass your virtual key saved on Portkey for any provider you'd like (Anthropic, OpenAI, Groq, etc.)
        provider="anthropic",
        # anthropic_beta="prompt-caching-2024-07-31", # to add cache, add "cache_control": {"type": "ephemeral"} in respective message body
    ),
    model="claude-3-5-sonnet-latest", #claude-3-5-haiku-20241022
    temperature=0.9,
    max_tokens = 150,
    max_retries=2,
)

# LANGGRAPH IMPLEMENTATION STARTS
# Define the state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Create the base Agent class
# # Base Agent class using Redis
# class Agent:
#     def __init__(self, prompt_template, model, redis_client, namespace):
#         self.prompt_template = prompt_template
#         self.model = model
#         self.redis_client = redis_client
#         self.namespace = namespace
#         # Clear the Redis list when a new object is created
#         self.redis_client.delete(self.namespace)

#     def run(self, state: State):
#         print(f"Running {self.__class__.__name__}")
#         # Retrieve the last 2 conversations from Redis
#         memories = self.redis_client.lrange(self.namespace, -2, -1)
#         memories = [memory.decode('utf-8') for memory in memories]
#         info = "\n".join(memories)
#         # Incorporate memories into the system message
#         system_msg = self.prompt_template.format(input="{input}") + f"\nTake into account these past conversations: {info}"
#         response = self.model.invoke(
#             [{"type": "system", "content": system_msg}] + state["messages"]
#         )
#         # Store the new conversation in Redis
#         memory = str(response)
#         self.redis_client.rpush(self.namespace, memory)
#         # Trim the list to keep only the last 2 conversations
#         self.redis_client.ltrim(self.namespace, -2, -1)
#         return {"messages": response}

class Agent:
    def __init__(self, prompt_template, model, store: BaseStore):
        self.prompt_template = prompt_template
        self.model = model
        self.store = store
        self.namespace = ("memories", "123")  # Using the same namespace across agents
        # Clear memory when a new object is created
        # for key in list(self.store.store.keys()):
        #     if key[0] == self.namespace[0] and key[1] == self.namespace[1]:
        #         del self.store.store[key]
    
    def run(self, state: State):
        print(f"Running {self.__class__.__name__}")
        # Retrieve the last 6 memories if available
        memories = self.store.search(self.namespace)
        last_memories = memories[-6:] if len(memories) > 6 else memories
        info = "\n".join([d.value["data"] for d in last_memories])
        # Incorporate memories into the system message
        system_msg = self.prompt_template.format(input="{input}") + f"\nTake into account these past conversations: {info}"
        response = self.model.invoke(
            [{"type": "system", "content": system_msg}] + state["messages"]
        )
        # Store new memories unconditionally or based on specific criteria
        # last_message = state["messages"][-1]
        # if "remember" in last_message.content.lower():
        memory = str(response)
        self.store.put(self.namespace, str(uuid.uuid4()), {"data": memory})
        return {"messages": response}

# DEFINING THE PROMPT

# Define mapping of prompt names to their corresponding prompt keys in the prompts dict
prompt_configs = {
    'planner': 'planner_agent_prompt',
    'conversational': 'empathetic_agent_prompt', 
    'suicide_prevention': 'suicide_prevention_agent_prompt',
    'anger_management': 'anger_prevention_agent_prompt',
    'motivational': 'motivational_agent_prompt',
    'dialectical_behavior_therapy': 'dbt_agent_prompt',
    'cognitive_behavioral_therapy': 'cbt_agent_prompt'
}

# Create prompts using a dictionary comprehension
prompts_dict = {
    f"{name}_prompt": ChatPromptTemplate.from_messages([
        ("system", prompts[prompt_key]),
        ("human", "{input}")
    ])
    for name, prompt_key in prompt_configs.items()
}

# Unpack the prompts into individual variables
planner_prompt = prompts_dict['planner_prompt']
conversational_prompt = prompts_dict['conversational_prompt']
suicide_prevention_prompt = prompts_dict['suicide_prevention_prompt']
anger_management_prompt = prompts_dict['anger_management_prompt']
motivational_prompt = prompts_dict['motivational_prompt']
dialectical_behavior_therapy_prompt = prompts_dict['dialectical_behavior_therapy_prompt']
cognitive_behavioral_therapy_prompt = prompts_dict['cognitive_behavioral_therapy_prompt']

# Define subclasses for each specific agent
class ConversationalAgent(Agent):
    pass  # Inherits from Agent
class SuicidePreventionAgent(Agent):
    pass  # Inherits from Agent
class AngerManagementAgent(Agent):
    pass  # Inherits from Agent
class MotivationalAgent(Agent):
    pass  # Inherits from Agent
class DialecticalBehaviorTherapyAgent(Agent):
    pass  # Inherits from Agent
class CognitiveBehavioralTherapyAgent(Agent):
    pass  # Inherits from Agent

# Initialize agent instances with their respective prompts and models
# Initialize agent instances with their respective prompts and models
conversational_agent = ConversationalAgent(conversational_prompt, model, redis_client, 'conversational_memories')
suicide_prevention_agent = SuicidePreventionAgent(suicide_prevention_prompt, model_a, redis_client, 'suicide_memories')
anger_management_agent = AngerManagementAgent(anger_management_prompt, model_a, redis_client, 'anger_memories')
motivational_agent = MotivationalAgent(motivational_prompt, model_a, redis_client, 'motivational_memories')
dbt_agent = DialecticalBehaviorTherapyAgent(dialectical_behavior_therapy_prompt, model_a, redis_client, 'dbt_memories')
cbt_agent = CognitiveBehavioralTherapyAgent(cognitive_behavioral_therapy_prompt, model_a, redis_client, 'cbt_memories')

def validate_input(message: str) -> bool:
    """
    Validates if the input is appropriate for mental health discussion.
    TODO: Implement comprehensive validation logic using:
    - Content moderation API
    - Keyword filtering
    - Sentiment analysis
    - Custom LLM classifier
    """
    # Placeholder for input validation
    # I'll add validation logic here
    inappropriate_keywords = {'hack', 'crack', 'exploit', 'illegal'}
    return not any(keyword in message.lower() for keyword in inappropriate_keywords)


# Define router
def route_query(state: State):
    class RouteQuery(BaseModel):
        route: Literal[
            "conversational", "suicide_prevention", "anger_management", 
            "motivational", "dialectical_behavior_therapy", "cognitive_behavioral_therapy"
        ] = Field(
            ...,
            description="Think step by step and direct to the most appropriate agent."
        )
    
    last_message = state["messages"][-1]
    
    # First check if input is valid
    # if not validate_input(last_message.content):
    #     return "invalid_input"
        
    structured_llm_router = model.with_structured_output(RouteQuery)
    question_router = planner_prompt | structured_llm_router
    resp = question_router.invoke({"input": last_message.content})
    return resp.route

# Create a custom graph class to handle agents
# Create a custom graph class to handle agents
class AgentStateGraph(StateGraph):
    def __init__(self, state_type):
        super().__init__(state_type)
        # Dictionary to hold agents
        self.agents = {
            "conversational": conversational_agent,
            "suicide_prevention": suicide_prevention_agent,
            "anger_management": anger_management_agent,
            "motivational": motivational_agent,
            "dialectical_behavior_therapy": dbt_agent,
            "cognitive_behavioral_therapy": cbt_agent
        }
    
    def add_agent_node(self, name: str):
        def agent_runner(state: State):
            if name == "invalid_input":
                return {"messages": [AIMessage(content="Please use this wisely. This space is for mental and emotional well-being. Namaste.")]}
            agent = self.agents[name]
            return agent.run(state)
        self.add_node(name, agent_runner)
# Instantiate the graph
workflow = AgentStateGraph(State)

# Add nodes for each agent
agent_names = [
    "conversational",
    "suicide_prevention",
    "anger_management",
    "motivational",
    "dialectical_behavior_therapy",
    "cognitive_behavioral_therapy",
    # "invalid_input"  # Added as a special routing case
]

for agent_name in agent_names:
    workflow.add_agent_node(agent_name)

# Add edges
workflow.add_conditional_edges(
    START,
    route_query,
    {name: name for name in agent_names}
)
for agent_name in agent_names:
    workflow.add_edge(agent_name, END)

# Compile the graph
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Function to run a conversation turn
def chat(message: str, config: dict):
    result = graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    return result["messages"][-1]

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Sukoon: Goodbye!")
            break
        response = chat(user_input, config)
        print("Sukoon:", response.content)