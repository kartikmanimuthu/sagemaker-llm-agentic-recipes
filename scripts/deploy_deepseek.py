# %% [markdown]
# ## Install Dependencies
# This section upgrades the SageMaker SDK to the latest version

# NOTE: Install SageMaker SDK before running this script:
# pip install 'sagemaker<3.0.0'

# %%

import sagemaker
import boto3
import os
# --- FIX --- Import Session specifically from sagemaker.session
from sagemaker.session import Session 


# --- Configuration Section ---
# Set these via environment variables before running
PROFILE_NAME = os.environ.get('AWS_PROFILE', 'default')
ROLE_ARN = os.environ.get('SAGEMAKER_ROLE_ARN')  # Required: Your SageMaker execution role ARN
HF_TOKEN = os.environ.get("HF_TOKEN")  # Required for gated models

# --- Session Initialization ---

# Create a Boto3 session using the specified local profile
boto_session = boto3.Session(profile_name=PROFILE_NAME)

# Create the SageMaker Session using the explicitly imported Session class
sess = Session(boto_session=boto_session)

# Use the explicitly defined Role ARN
role = ROLE_ARN 

print(f"AWS Region: {sess.boto_region_name}")
print(f"SageMaker Execution Role: {role}")

# %%
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

# --- Model Configuration ---

# NOTE: Using a smaller model for this example, which is suitable for text classification.
MODEL_ID = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
TASK = 'text-generation'
INSTANCE_TYPE = "ml.g5.2xlarge" # 1x A10G GPU (24GB VRAM) - Supports BF16 and sufficient for 8B model


hub = {
    'HF_MODEL_ID': MODEL_ID, 
    'HF_TASK': TASK,
    # Pass the token to the container only if needed for private models
    'HF_TOKEN': HF_TOKEN,
    'HF_MODEL_DEVICE_MAP': 'auto', # Distribute model across all available GPUs
    'SM_NUM_GPUS': '1', # Explicitly set number of GPUs for the container
    'HF_TRUST_REMOTE_CODE': 'True' # Allow custom model architectures 
}

# --- Create HuggingFaceModel Object ---

# You must choose compatible versions for the container. 
# Check the Hugging Face SageMaker documentation for the latest versions.
huggingface_model = HuggingFaceModel(
    env=hub,
    role=role,                 
    image_uri=get_huggingface_llm_image_uri("huggingface",version="2.2.0"),
    sagemaker_session=sess,       # Use the session with the correct profile
)

# --- Deploy the Model ---

print(f"Starting deployment of {MODEL_ID} to endpoint...")

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type=INSTANCE_TYPE,
    container_startup_health_check_timeout=300
)

print(f"Deployment complete. Endpoint Name: {predictor.endpoint_name}")

# %%
# --- Inference Test Data ---
data = {
    "inputs": "what is aws sagemaker"
}

# --- Invoke the Endpoint ---
print("Invoking the endpoint...")
response = predictor.predict(data)

# --- Print Results ---
print("\n--- Prediction Response ---")
print(response[0]['generated_text'])
print("---------------------------\n")

# [{'generated_text': '...'}]

# %%
# --- Delete the Endpoint ---
print(f"Deleting endpoint: {predictor.endpoint_name}...")
predictor.delete_endpoint()
print("Endpoint deleted successfully.")


# %%
