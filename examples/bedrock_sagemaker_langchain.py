# %% [markdown]
# # SageMaker Bedrock Integration via LangChain
# 
# This notebook demonstrates how to invoke a "Bedrock Ready" SageMaker endpoint using the standard `ChatBedrock` class from `langchain_aws`.
# 
# **Prerequisites:**
# 1. You must have a deployed SageMaker endpoint.
# 2. You must have "Registered" that endpoint with Amazon Bedrock to get an ARN.
#    - Go to SageMaker or Bedrock console -> Register with Bedrock -> Copy the ARN.

# %%
# Install dependencies
# We need boto3, langchain, langchain-aws, and langgraph
import sys
import subprocess

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3", "langchain", "langchain-aws", "langgraph", "-q"])
except subprocess.CalledProcessError as e:
    print(f"Warning: Dependency installation failed: {e}")

# %%
import boto3
import os
from langchain_aws import ChatBedrock
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

# %% [markdown]
# ## 1. Configuration
# Paste your Bedrock-Registered Endpoint ARN below.

# %%
# TODO: REPLACE THIS WITH YOUR ACTUAL ARN FROM BEDROCK REGISTRATION
# Format: arn:aws:bedrock:<region>:<account>:imported-model/<model-id>

# CUSTOM_MODEL_ARN = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/jumpstart-dft-deepseek-llm-r1-disti-20251206-121042" 
CUSTOM_MODEL_ARN = "arn:aws:bedrock:us-east-1:123456789012:imported-model/zv2kzta4g1ug" 
PROFILE_NAME = os.environ.get('AWS_PROFILE', 'default')
REGION_NAME = "us-east-1" # Ensure this matches your SageMaker/Bedrock region

print(f"Using Model ARN: {CUSTOM_MODEL_ARN}")

# %%
# ## 2. Initialize ChatBedrock
# We use the standard ChatBedrock class, passing the ARN as the model_id.

# Setup Boto3 Client
try:
    boto_session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)
    bedrock_runtime = boto_session.client("bedrock-runtime")
    print(f"Authenticated with profile: {PROFILE_NAME}")
except Exception as e:
    print(f"Failed to use profile {PROFILE_NAME}. Using default credentials.")
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION_NAME)

# Initialize LLM
# Initialize LLM
# Note: When using an ARN (Imported Model), 'provider' argument is mandatory.
# "sagemaker" is not a valid Chat provider in LangChain (it's for text-generation only).
# We use "meta" (Llama 2/3) as the provider because DeepSeek models generally follow 
# Llama-compatible chat formatting/templates. If formatting issues arise, 
# you may need to implement a custom wrapper.
llm = ChatBedrock(
    model_id=CUSTOM_MODEL_ARN,
    client=bedrock_runtime,
    provider="meta", 
    model_kwargs={
        "max_tokens": 1024,
        "temperature": 0.7,
        # "stop": ["<|eot_id|>"] # Uncomment if model needs specific stop tokens
    }
)

# Simple Test
try:
    print("Running simple invocation test...")
    # Note: If the ARN is invalid, this will fail.
    response = llm.invoke("What is aws sagemaker?")
    print(response.content)
except Exception as e:
    print(f"Error during test: {e}")

# %% [markdown]
# ## 3. Interactive Chat (Simple LangChain)
# Using a simple loop without LangGraph overhead.

# %%
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def chat():
    print("Starting Chat (type 'quit' to exit)...")
    if "your-model-id" in CUSTOM_MODEL_ARN:
         print("WARNING: You have not replaced the CUSTOM_MODEL_ARN with your actual ARN yet.")
    
    # Store history as LangChain Message objects
    history = []
    history.append(SystemMessage(content="You are a helpful assistant! Your name is Bob.")) 
    
    while True:
        user_input = input("User: ")
        # print(f"User: {user_input}") # Optional debug print
        if user_input.lower() in ['quit', 'exit']:
            break

        
        # Add user message to history
        history.append(HumanMessage(content=user_input))
        
        try:
            print("Assistant parsing...")
            # Invoke LLM with full history
            response = llm.invoke(history)
            content = response.content
            
            # Print response
            print(f"Assistant: {content}")
            
            # Add assistant response to history
            history.append(AIMessage(content=content))
            
        except Exception as e:
            print(f"Error: {e}")

chat() # Uncomment to run

# %%
