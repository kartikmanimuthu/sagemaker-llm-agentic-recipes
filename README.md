# SageMaker Agentic LLM Recipes 🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange.svg)](https://aws.amazon.com/sagemaker/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Integration-blueviolet.svg)](https://langchain-ai.github.io/langgraph/)

**SageMaker Agentic LLM Recipes** is a curated collection of production-grade patterns for deploying Open Source LLMs on AWS SageMaker and orchestrating them into intelligent agents using **LangGraph**.

This repository goes beyond simple inference—it demonstrates how to build **stateful, reasoning agents** that leverage the scale of SageMaker and the cognitive architecture of LangGraph.

---

## 🏗️ Project Structure

The repository is organized into focused areas to help you find exactly what you need:

- **[`notebooks/`](./notebooks)**: Interactive deployment and inference guides for various models.
- **[`scripts/`](./scripts)**: Production-ready Python scripts for automation, endpoint cleanup, and benchmarking.
- **[`examples/`](./examples)**: **Agentic workflows** and advanced integration patterns including LangGraph, Bedrock, and multi-model orchestration.

## 📊 Model Support Matrix

| Family | Model | Type | Recipe Status |
| :--- | :--- | :--- | :--- |
| **DeepSeek** | DeepSeek-V3 / R1 | LLM | :white_check_mark: |
| **Meta** | Llama 2 / 3 | LLM | :white_check_mark: |
| **Mistral AI** | Mistral-7B | LLM | :white_check_mark: |
| **Google** | Gemma | LLM | :white_check_mark: |
| **OpenAI (OSS)** | GPT-NeoX 20B | LLM | :white_check_mark: |

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/kartikmanimuthu/sagemaker-llm-recipes.git
cd sagemaker-llm-recipes
```

### 2. Install Dependencies
We recommend using a virtual environment (Python 3.10+):
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Set required environment variables:
```bash
# AWS Profile (if not using default)
export AWS_PROFILE=your-profile

# SageMaker Execution Role ARN (required)
export SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/YOUR_SAGEMAKER_ROLE

# HuggingFace Token (required for gated models like Gemma, Llama)
export HF_TOKEN=your-huggingface-token
```

### 4. Deploy Your First Model
```bash
cd scripts
python deploy_deepseek.py
```

---

## 🎯 Why SageMaker for Agentic AI?

### 1. The Foundation for Agents
Agents require fast, reliable inference to "think" and "act" in loops. SageMaker provides the dedicated compute (GPUs) needed for low-latency reasoning steps, which is critical for complex LangGraph workflows.

### 2. Zero to Production in Minutes
SageMaker eliminates the operational complexity typically associated with deploying large language models. With our recipes, you can:
- **Skip Infrastructure Management**: No need to provision GPUs, configure CUDA, or manage container orchestration
- **Direct HuggingFace Integration**: Deploy any open-source model from HuggingFace Hub with a few lines of code
- **Auto-Scaling Out of the Box**: Built-in load balancing handles traffic spikes from agent swarms
- **Pay Only For What You Use**: Shut down endpoints when not needed, use Spot instances for 70% cost savings

### 3. Orchestrate with LangGraph
This repository showcases how to move beyond "stateless" calls. using LangGraph, you can:
- **Maintain State**: Keep conversation history and context across multi-step reasoning
- **Control Flow**: Define loops, branches, and conditional logic for your agents
- **Integrate Tools**: Connect SageMaker LLMs to external APIs and databases

### The SageMaker Advantage

| Traditional Deployment | SageMaker Deployment |
| :--- | :--- |
| Manual EC2 + GPU setup | Managed infrastructure |
| Stateless API calls | **Stateful Agentic Workflows** |
| Custom container builds | Pre-built HuggingFace containers |
| DIY monitoring & logging | Integrated CloudWatch metrics |
| Days to production | Minutes to production |

### Perfect for OSS LLM Experimentation
SageMaker's managed approach makes it ideal for:
- **Rapid Prototyping**: Test multiple model architectures without DevOps overhead
- **Research & Development**: Focus on model evaluation, not infrastructure
- **Production Inference**: Scale from prototype to production without rewriting code

## 💡 Key Features

- **Cost Optimization**: Recipes optimized for Spot Instances and efficient `ml.g5` instance selection.
- **Security First**: Dynamic IAM role retrieval and zero-hardcoded-credential patterns.
- **Orchestration**: Seamless integration with **LangGraph** for building stateful, multi-step AI agents.

## 📖 Detailed Usage

For comprehensive deployment patterns, configuration options, and troubleshooting, see [`USAGE.md`](./USAGE.md).

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place! Please see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for guidelines on how to add new model recipes.

## 🛡️ Security

For vulnerability reporting and security best practices, please refer to our [`SECURITY.md`](./SECURITY.md).

## 📄 License

Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.

---
*Maintained with ❤️ by [Kartik](https://github.com/kartikmanimuthu)*
