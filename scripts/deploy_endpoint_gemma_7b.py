# %% [markdown]
# ## Install Dependencies
# This section upgrades the SageMaker SDK to the latest version compatible with the script

########################################################################

# %%
# NOTE: Install SageMaker SDK before running this script:
# pip install 'sagemaker<3.0.0'

########################################################################
# %%

import sagemaker
import boto3
import os
import json
# --- FIX --- Import Session specifically from sagemaker.session
from sagemaker.session import Session 
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri, HuggingFacePredictor


# --- Configuration Section ---
# Set these via environment variables before running
PROFILE_NAME = os.environ.get('AWS_PROFILE', 'default')
ROLE_ARN = os.environ.get('SAGEMAKER_ROLE_ARN')  # Required: Your SageMaker execution role ARN
HF_TOKEN = os.environ.get("HF_TOKEN")  # Required for gated models like Gemma

# --- Session Initialization ---

# Create a Boto3 session using the specified local profile
try:
    boto_session = boto3.Session(profile_name=PROFILE_NAME)
except Exception as e:
    print(f"Profile {PROFILE_NAME} not found, using default credentials.")
    boto_session = boto3.Session()

# Create the SageMaker Session using the explicitly imported Session class
sess = Session(boto_session=boto_session)

# Use the explicitly defined Role ARN
role = ROLE_ARN 

print(f"AWS Region: {sess.boto_region_name}")
print(f"SageMaker Execution Role: {role}")


########################################################################

# %%

# --- Model Configuration ---

# IMPORANT: google/gemma-7b is a GATED model. 
# You MUST accept the license terms at https://huggingface.co/google/gemma-7b 
# and use a Hugging Face token that has read access to this model.
MODEL_ID = 'google/gemma-7b'
# Using ml.g5.12xlarge (4 GPUs) as per snippet request. 
# Note: ml.g5.2xlarge (1 GPU) is also sufficient for 7B models if cost is a concern.
INSTANCE_TYPE = "ml.g5.2xlarge" 
NUM_GPUS = 1
ENDPOINT_NAME = 'gemma-7b-inference-optimized-1'

hub = {
    'HF_MODEL_ID': MODEL_ID,
    'SM_NUM_GPUS': json.dumps(NUM_GPUS),
    'HF_TOKEN': HF_TOKEN,
    'HF_TASK': 'text-generation',
    'HF_MODEL_DEVICE_MAP': 'auto',
    'HF_TRUST_REMOTE_CODE': 'True',
    # Reduce token limits to avoid OOM on ml.g5.2xlarge (24GB VRAM) due to large vocab (256k)
    'MAX_BATCH_PREFILL_TOKENS': '1024',
    'MAX_INPUT_TOKENS': '1024',
    'MAX_TOTAL_TOKENS': '2048'
}

# --- Create HuggingFaceModel Object ---

# Using version 2.2.0 which supports Gemma and is compatible with the SDK
image_uri = get_huggingface_llm_image_uri("huggingface", version="2.2.0")

huggingface_model = HuggingFaceModel(
    image_uri=image_uri, 
    env=hub,
    role=role, 
    sagemaker_session=sess
)

# --- Deploy the Model ---

print(f"Checking for existing endpoint: {ENDPOINT_NAME}...")
sm_client = boto_session.client("sagemaker")
endpoint_exists = False

try:
    response = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    status = response['EndpointStatus']
    print(f"Endpoint found with status: {status}")
    if status in ['InService', 'Creating', 'Updating']:
        endpoint_exists = True
except Exception:
    print("Endpoint not found or error checking status.")

if endpoint_exists:
    print(f"Skipping deployment. Attaching to existing endpoint: {ENDPOINT_NAME}")
    predictor = HuggingFacePredictor(
        endpoint_name=ENDPOINT_NAME,
        sagemaker_session=sess
    )
else:
    print(f"Starting deployment of {MODEL_ID} to endpoint: {ENDPOINT_NAME}...")
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        container_startup_health_check_timeout=3600,
        endpoint_name=ENDPOINT_NAME
    )
    print(f"Deployment complete. Endpoint Name: {predictor.endpoint_name}")

########################################################################

# %%
# --- Inference Test Data ---
data = {
    "inputs": "My name is Julien and I like to",
    "parameters": {
        "max_new_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
    }
}

# --- Invoke the Endpoint ---
print("Invoking the endpoint...")
try:
    response = predictor.predict(data)
    
    # --- Print Results ---
    print("\n--- Prediction Response ---")
    # Handle list or dict response
    if isinstance(response, list) and len(response) > 0:
         print(response[0].get('generated_text', response))
    else:
        print(response)
    print("---------------------------\n")

except Exception as e:
    print(f"Error invoking endpoint: {e}")

########################################################################

# %%
# --- Delete the Endpoint (Optional) ---
# Uncomment the following lines to delete the endpoint after testing
print(f"Deleting endpoint: {predictor.endpoint_name}...")
predictor.delete_endpoint()
print("Endpoint deleted successfully.")
