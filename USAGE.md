# Usage Guide

This guide provides detailed instructions on using SageMaker LLM Recipes to deploy open-source language models from HuggingFace.

## Prerequisites

### 1. AWS Account Setup
- An active AWS account with SageMaker access
- Appropriate service quotas for GPU instances (e.g., `ml.g5.2xlarge`, `ml.g5.12xlarge`)
- A SageMaker execution role with necessary permissions

### 2. Create SageMaker Execution Role
If you don't have one, create a role via AWS Console or CLI:

```bash
# Create role with AmazonSageMakerFullAccess policy
aws iam create-role --role-name SageMakerExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach required policy
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

### 3. HuggingFace Access Token
For gated models (Llama, Gemma, etc.):
1. Create a HuggingFace account at https://huggingface.co/
2. Accept model license agreements (e.g., https://huggingface.co/meta-llama/Llama-2-7b-hf)
3. Generate an access token at https://huggingface.co/settings/tokens

---

## Deployment Patterns

### Pattern 1: Quick Deployment (Scripts)

For rapid one-off deployments using Python scripts:

```bash
# Set environment variables
export SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/YOUR_ROLE
export HF_TOKEN=your_huggingface_token
export AWS_PROFILE=your-aws-profile  # Optional

# Deploy DeepSeek model
cd scripts
python deploy_deepseek.py
```

**What happens:**
- Creates a SageMaker endpoint with the specified model
- Performs a test inference
- Deletes the endpoint (optional, comment out cleanup code to keep it)

### Pattern 2: Interactive Exploration (Notebooks)

For experimenting with different configurations:

```bash
# Launch Jupyter
jupyter notebook notebooks/

# Open and run any notebook, e.g.:
# - 01_deploy_openai_gpt_oss_20b.ipynb
# - 03_deploy_huggingface_to_sagemaker.ipynb
```

**Benefits:**
- Step-by-step execution with inline documentation
- Easy parameter tuning (instance types, model IDs, etc.)
- Visual feedback on each step

### Pattern 3: Advanced Orchestration (Examples)

For building LLM-powered applications with LangGraph:

```bash
cd examples

# Run LangGraph chat example
python sagemaker_chat_langgraph.py
```

---

## Configuration Options

### Instance Types

Choose based on your model size and budget:

| Instance Type | GPUs | VRAM | Suitable For | Cost (On-Demand) |
| :--- | :--- | :--- | :--- | :--- |
| `ml.g5.xlarge` | 1x A10G | 24 GB | 7B models, testing | ~$1/hour |
| `ml.g5.2xlarge` | 1x A10G | 24 GB | 7B-8B models | ~$2/hour |
| `ml.g5.12xlarge` | 4x A10G | 96 GB | 13B-20B models | ~$7/hour |
| `ml.g5.48xlarge` | 8x A10G | 192 GB | 70B+ models | ~$16/hour |

> **Tip:** Use Spot instances to save up to 70% on compute costs.

### Environment Variables

Required variables for all scripts:

```bash
# Required
export SAGEMAKER_ROLE_ARN=arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME

# Optional (uses 'default' profile if not set)
export AWS_PROFILE=your-profile-name

# Required only for gated models
export HF_TOKEN=hf_...
```

---

## Model Selection

All models are pulled directly from HuggingFace Hub. Simply change the `MODEL_ID`:

```python
# In any script or notebook
MODEL_ID = 'meta-llama/Llama-2-7b-hf'        # Llama 2
MODEL_ID = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'  # DeepSeek
MODEL_ID = 'google/gemma-7b'                 # Gemma (requires license)
MODEL_ID = 'mistralai/Mistral-7B-v0.1'       # Mistral
```

---

## Cost Management

### 1. Delete Endpoints When Not in Use
```python
predictor.delete_endpoint()
```

### 2. Use Async Endpoints for Batch Processing
For non-real-time workloads, use asynchronous inference to reduce costs.

### 3. Monitor with CloudWatch
Set billing alarms to avoid unexpected charges:
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name sagemaker-billing-alert \
  --metric-name EstimatedCharges \
  --namespace AWS/SageMaker \
  --threshold 100
```

---

## Troubleshooting

### Issue: "ResourceLimitExceeded"
**Cause:** Insufficient service quota for the instance type.

**Solution:** Request a quota increase in AWS Service Quotas console:
1. Go to Service Quotas → Amazon SageMaker
2. Search for your instance type (e.g., `ml.g5.2xlarge`)
3. Request an increase

### Issue: "ModelError: Loading model failed"
**Cause:** Model requires authentication (gated model) and `HF_TOKEN` is missing.

**Solution:** Ensure `HF_TOKEN` environment variable is set and you've accepted the model's license.

### Issue: Slow inference
**Cause:** Model loaded on CPU instead of GPU, or insufficient VRAM.

**Solution:** 
- Verify `SM_NUM_GPUS` is set correctly
- Use a larger instance type with more VRAM

---

## What Makes This Easy?

Traditional LLM deployment requires:
- ✗ Manual GPU instance provisioning
- ✗ CUDA/Docker configuration
- ✗ Load balancer setup
- ✗ Monitoring infrastructure
- ✗ Custom scaling logic

**With SageMaker:**
- ✓ One command deployment
- ✓ Pre-configured HuggingFace containers
- ✓ Built-in load balancing
- ✓ Integrated CloudWatch
- ✓ Auto-scaling out of the box

**Result:** Focus on model experimentation, not DevOps.
