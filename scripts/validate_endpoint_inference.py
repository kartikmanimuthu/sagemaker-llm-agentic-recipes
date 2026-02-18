# %% [markdown]
# ## AWS Sage Maker Endpoint Inferencing
# This section upgrades the SageMaker SDK to the latest version


# %%
# --- Configuration ---
# Use the same profile as in the deployment script
PROFILE_NAME = os.environ.get('AWS_PROFILE', 'default') 
# Replace this with the actual endpoint name from your deployment output
ENDPOINT_NAME = 'huggingface-pytorch-tgi-inference-2025-12-05-07-20-36-989' 
REGION_NAME = 'ap-south-1' # Ensure this matches your deployment region


# %%
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFacePredictor
from sagemaker.session import Session
import os

# --- Session Initialization ---
try:
    print(f"Initializing session with profile: {PROFILE_NAME}")
    boto_session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)
    sess = Session(boto_session=boto_session)
except Exception as e:
    print(f"Failed to use profile {PROFILE_NAME}: {e}")
    print("Falling back to default credentials...")
    boto_session = boto3.Session(region_name=REGION_NAME)
    sess = Session(boto_session=boto_session)

print(f"AWS Region: {sess.boto_region_name}")

# %%

# --- Connect to Existing Endpoint ---
print(f"Attaching to endpoint: {ENDPOINT_NAME}")

# Create a predictor object attached to the existing endpoint
predictor = HuggingFacePredictor(
    endpoint_name=ENDPOINT_NAME,
    sagemaker_session=sess
)

# --- Inference Parameters ---
# These parameters control the generation behavior
parameters = {
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}

# %%

# --- Test Input ---
input_text = "Explain the benefits of using AWS SageMaker for machine learning."
payload = {
    "inputs": input_text,
    "parameters": parameters
}

# --- Run Inference ---
print(f"\nSending request with input: '{input_text}'")
print("Waiting for response...")

try:
    response = predictor.predict(payload)
    
    print("\n--- Prediction Response ---")
    # The response from TGI is typically a list of dictionaries
    if isinstance(response, list) and len(response) > 0:
        generated_text = response[0].get('generated_text', response)
        print(generated_text)
    else:
        print(response)
    print("---------------------------\n")

except Exception as e:
    print(f"\nError during inference: {e}")
    print("Possible causes:")
    print("1. The endpoint is not InService (check SageMaker console).")
    print("2. The payload format is incorrect for the model.")
    print("3. Network/Permissions issues.")

# %%
