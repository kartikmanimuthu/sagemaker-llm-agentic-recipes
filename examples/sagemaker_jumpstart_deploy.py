# %% [markdown]
# # SageMaker JumpStart Deployment & LangGraph Chat
# 
# This notebook demonstrates how to:
# 1. Deploy the `deepseek-llm-r1-distill-qwen-1-5b` model using SageMaker JumpStart.
# 2. Authenticate using a specific AWS Profile.
# 3. Integrate the deployed endpoint with LangGraph for stateful chat.

# %%
# Install dependencies
# pip install -r requirements.txt -q
# pip install 'sagemaker==2.251.1' --no-deps

# ## 1. Setup & Authentication
# We configure the AWS session using the specified profile.

import sagemaker
import boto3
from sagemaker.jumpstart.model import JumpStartModel
import json
import time
from botocore.exceptions import ClientError

def get_or_create_sagemaker_role(role_name, boto_session):
    iam = boto_session.client("iam")
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    role_arn = None
    try:
        response = iam.get_role(RoleName=role_name)
        role_arn = response["Role"]["Arn"]
        print(f"Role {role_name} exists. Verifying trust policy...")
        
        # Enforce Trust Policy
        iam.update_assume_role_policy(
            RoleName=role_name,
            PolicyDocument=json.dumps(trust_policy)
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            print(f"Creating new role: {role_name}")
            response = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="SageMaker execution role created by JumpStart notebook"
            )
            role_arn = response["Role"]["Arn"]
        else:
            print(f"Error checking role: {e}")
            raise
    
    # Always ensure permissions are attached
    try:
        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
        )
        print(f"Ensured AmazonSageMakerFullAccess is attached to {role_name}")
    except Exception as e:
        print(f"Warning: Could not attach AmazonSageMakerFullAccess: {e}")
        
    print("Waiting 10 seconds for IAM role propagation...")
    time.sleep(10)
            
    return role_arn

# Configuration
# Configuration
PROFILE_NAME = os.environ.get('AWS_PROFILE', 'default')
REGION_NAME = 'us-east-1' # Change to 'ap-south-1' or other regions as needed
MODEL_ID = "deepseek-llm-r1-distill-qwen-1-5b"
MODEL_VERSION = "2.20.0"
ROLE_NAME = 'SageMakerExecutionRole-JumpStart-Deepseek'

# Establish Session
try:
    boto_session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    print(f"Authenticated with profile: {PROFILE_NAME} in region: {REGION_NAME}")
except Exception as e:
    print(f"Failed to use profile {PROFILE_NAME}. Falling back to default credentials.")
    boto_session = boto3.Session(region_name=REGION_NAME)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    print(f"Authenticated with default credentials in region: {REGION_NAME}")

# Setup IAM Role
try:
    ROLE_ARN = get_or_create_sagemaker_role(ROLE_NAME, boto_session)
    print(f"Using SageMaker Execution Role: {ROLE_ARN}")
except Exception as e:
    print(f"Failed to setup IAM role: {e}")
    # Fallback or exit if crucial
    raise e

# %% [markdown]
# ## 2. Deploy JumpStart Model
# We define the JumpStart model and deploy it. 
# **Note**: You must accept the EULA if this is a gated model (`accept_eula=True`).

# %%
# Define the model
# We pass the sagemaker_session to ensure it uses the correct profile
model = JumpStartModel(
    model_id=MODEL_ID,
    model_version=MODEL_VERSION,
    role=ROLE_ARN,
    sagemaker_session=sagemaker_session
)

# Deploy the endpoint
# accept_eula=True is often required for JumpStart models
try:
    predictor = model.deploy(accept_eula=True)
    ENDPOINT_NAME = predictor.endpoint_name
    print(f"Model deployed successfully! Endpoint Name: {ENDPOINT_NAME}")
except Exception as e:
    print(f"Deployment failed or endpoint already exists: {e}")
    # Attempt to retrieve existing endpoint if deployment fails (e.g. if you ran this cell twice)
    # Note: Logic to find existing endpoint by name would go here if needed.

# %% [markdown]
# ## 3. Basic Invoke Test
# Test the endpoint with a simple payload.

# %%
payload = {
    "inputs": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is Amazon SageMaker?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "parameters": {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9
    }
}

if 'predictor' in locals():
    response = predictor.predict(payload)
    print(response)

# %% [markdown]
# ## 4. LangGraph Integration
# Now we wrap the endpoint in a LangGraph `call_model` node.

# %%
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
import json

# Define Graph State
class State(TypedDict):
    messages: List[Dict[str, str]]

def call_model(state: State):
    messages = state["messages"]
    
    # Prepare payload
    # Using standard Chat API format if supported, or formatting manually.
    # Here assuming the model supports 'inputs' string prompts or chat-formatted inputs.
    
    # Manual formatting for DeepSeek/Llama-3 style
    prompt = "<|begin_of_text|>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["<|eot_id|>"]
        }
    }
    
    try:
        response = predictor.predict(payload)
        
        # Parse Logic
        if isinstance(response, bytes):
            response_data = json.loads(response.decode('utf-8'))
        else:
            response_data = response
            
        content = None
        
        # Handle [{'generated_text': '...'}]
        if isinstance(response_data, list) and len(response_data) > 0:
            item = response_data[0]
            full_text = item.get('generated_text')
            if full_text:
                # Strip prompt if echoed
                if full_text.startswith(prompt):
                    content = full_text[len(prompt):].strip()
                else:
                    content = full_text.strip()
        elif isinstance(response_data, dict) and 'generated_text' in response_data:
             content = response_data['generated_text']

        if not content:
            content = "Error: No content generated."
            
        return {"messages": messages + [{"role": "assistant", "content": content}]}
        
    except Exception as e:
        return {"messages": messages + [{"role": "assistant", "content": f"Error: {str(e)}"}]}

# Build Graph
workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)
app = workflow.compile()

# %%
# Interactive Chat
def chat():
    print("Starting Chat (type 'quit' to exit)...")
    history = []
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        
        history.append({"role": "user", "content": user_input})
        output = app.invoke({"messages": history})
        history = output["messages"]
        print(f"Assistant: {history[-1]['content']}")

chat() # Uncomment to run

# %%
# ## 5. Cleanup Resources
# CAUTION: This will delete the endpoint, model, and the IAM role created earlier.

def cleanup_resources(predictor, role_name, boto_session):
    print(f"Starting cleanup for {role_name} and endpoint...")
    
    # 1. Delete Model
    # We delete the model first because deleting the endpoint might remove the endpoint config,
    # which the SDK sometimes relies on to find the model name.
    try:
        if predictor:
            print(f"Deleting model: {predictor.endpoint_name}") 
            predictor.delete_model()
            print("Model deleted.")
    except Exception as e:
        print(f"Error deleting model (or model already deleted): {e}")

    # 2. Delete Endpoint
    try:
        if predictor:
            print(f"Deleting endpoint: {predictor.endpoint_name}")
            predictor.delete_endpoint()
            print("Endpoint deleted.")
    except Exception as e:
        print(f"Error deleting endpoint (or endpoint already deleted): {e}")

    # 3. Delete IAM Role
    iam = boto_session.client("iam")
    try:
        # Detach all policies first
        attached_policies = iam.list_attached_role_policies(RoleName=role_name)
        for policy in attached_policies['AttachedPolicies']:
            print(f"Detaching policy: {policy['PolicyArn']}")
            iam.detach_role_policy(RoleName=role_name, PolicyArn=policy['PolicyArn'])
            
        # Delete Inline Policies if any (optional, but good practice)
        inline_policies = iam.list_role_policies(RoleName=role_name)
        for policy_name in inline_policies['PolicyNames']:
            print(f"Deleting inline policy: {policy_name}")
            iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)

        # Delete the Role
        print(f"Deleting role: {role_name}")
        iam.delete_role(RoleName=role_name)
        print(f"Role {role_name} deleted successfully.")
        
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchEntity":
            print(f"Role {role_name} does not exist.")
        else:
            print(f"Error cleaning up role: {e}")
    except Exception as e:
        print(f"Unexpected error during role cleanup: {e}")

# Uncomment the following line to run cleanup
# cleanup_resources(predictor, ROLE_NAME, boto_session)
# %%
