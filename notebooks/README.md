# Notebooks

Interactive Jupyter notebooks for exploring SageMaker LLM deployments.

## Overview

These notebooks provide step-by-step guides for deploying various open-source language models using AWS SageMaker. Each notebook is self-contained with inline documentation explaining the concepts and configurations.

## Available Notebooks

### 01_deploy_openai_gpt_oss_20b.ipynb
Deploy GPT-NeoX 20B, a large-scale open-source language model. Demonstrates:
- Working with 20B parameter models
- Multi-GPU deployment strategies
- Memory optimization techniques

### 02_inference_openai_gpt_oss_20b.ipynb
Perform inference on the deployed GPT-NeoX model. Covers:
- Endpoint invocation patterns
- Parameter tuning (temperature, top_p, max_tokens)
- Response parsing and handling

### 03_deploy_huggingface_to_sagemaker.ipynb
General-purpose guide for deploying any HuggingFace model to SageMaker. Includes:
- Model selection from HuggingFace Hub
- Container image configuration
- Instance type recommendations

### 04_sagemaker_huggingface_deployment.ipynb
Advanced deployment patterns with custom configurations:
- Custom inference code
- Environment variable management
- Production deployment best practices

### 05_sagemaker_chat_langgraph.ipynb
Integrate deployed models with LangGraph for conversational AI:
- Building chat interfaces
- Stateful conversation management
- Multi-turn dialogue handling

### 06_sagemaker_jumpstart_deploy_and_chat.ipynb
Use SageMaker JumpStart for one-click model deployment:
- Pre-configured model catalog
- Simplified deployment workflow
- Quick experimentation setup

## Usage

```bash
# Start Jupyter in this directory
jupyter notebook

# Or run individual notebooks with papermill
papermill 01_deploy_openai_gpt_oss_20b.ipynb output.ipynb
```

## Why Notebooks?

Notebooks are ideal for:
- **Learning**: Step-by-step execution with immediate feedback
- **Experimentation**: Easily modify parameters and re-run cells
- **Documentation**: Inline markdown cells explain each concept
- **Prototyping**: Rapid iteration without writing full scripts
