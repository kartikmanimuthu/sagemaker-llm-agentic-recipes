# Scripts

Production-ready Python scripts for automated model deployment and management.

## Overview

These scripts provide command-line interfaces for common SageMaker operations. Unlike notebooks, they're designed for:
- Automated deployments via CI/CD pipelines
- Batch processing
- Infrastructure as Code (IaC) workflows

## Available Scripts

### deploy_deepseek.py
Deploy DeepSeek R1 Distill (8B) model from HuggingFace:
```bash
export SAGEMAKER_ROLE_ARN=arn:aws:iam::ACCOUNT:role/ROLE
export HF_TOKEN=your_token
python deploy_deepseek.py
```

**What it does:**
- Creates SageMaker endpoint with DeepSeek-R1-Distill-Llama-8B
- Configures GPU acceleration and model device mapping
- Tests inference with sample prompt
- Optionally cleans up endpoint

### deploy_gemma.py
Deploy Google Gemma 7B (gated model):
```bash
export SAGEMAKER_ROLE_ARN=arn:aws:iam::ACCOUNT:role/ROLE
export HF_TOKEN=your_token  # Must have Gemma license access
python deploy_gemma.py
```

**What it does:**
- Checks for existing endpoints to avoid duplicates
- Deploys Gemma-7B with optimized token limits
- Handles gated model authentication
- Performs test inference

### delete_all_endpoints.py
Clean up all SageMaker endpoints in your account:
```bash
python delete_all_endpoints.py
```

**Use case:** Cost management by removing unused endpoints.

> ⚠️ **Warning:** This will delete ALL endpoints. Use with caution in production accounts.

### test_local_inference.py
Test model inference locally before SageMaker deployment:
```bash
python test_local_inference.py
```

**Benefits:**
- Faster iteration without deployment overhead
- Debug model loading issues
- Validate tokenization and generation parameters

## Environment Variables

All scripts require these environment variables:

```bash
# Required
export SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/YOUR_ROLE

# Optional (defaults to 'default')
export AWS_PROFILE=your-profile

# Required for gated models only
export HF_TOKEN=hf_...
```

## Why Scripts Over Notebooks?

Scripts are better suited for:
- **Automation**: Run in CI/CD pipelines without manual intervention
- **Version Control**: Easier to track changes via Git
- **Productionization**: Deploy to cron jobs, Lambda functions, etc.
- **Repeatability**: Consistent execution environment
