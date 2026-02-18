# Agentic Examples

Advanced integration patterns combining SageMaker-deployed models with **LangGraph** for building reasoning agents.

## Overview

These examples demonstrate how to build **production-grade Agentic AI** by integrating SageMaker endpoints with orchestration frameworks. Unlike simple chatbots, these agents can maintain state, reason through problems, and execute multi-step workflows.

## Available Examples

### agent_stateful_chat_langgraph.py
Build a stateful conversational agent using LangGraph:

```bash
python agent_stateful_chat_langgraph.py
```

**Demonstrates:**
- Creating a LangGraph workflow with SageMaker endpoints
- Managing conversation state and memory
- Implementing multi-turn dialogue
- Claude-style thinking traces (for reasoning models)

**Use cases:**
- Customer support chatbots
- Research assistants
- Interactive tutorials

### rag_hybrid_bedrock_sagemaker.py
Hybrid architecture using both Bedrock and SageMaker:

```bash
python rag_hybrid_bedrock_sagemaker.py
```

**Demonstrates:**
- Combining AWS Bedrock models with custom SageMaker endpoints
- LangChain integration patterns
- Multi-model orchestration
- Cost optimization via model routing

**Use cases:**
- Using Bedrock for general tasks, SageMaker for specialized fine-tuned models
- A/B testing between Bedrock and open-source models
- Fallback patterns (primary + backup model)

### workflow_jumpstart_sdk_deploy.py
Programmatic deployment using SageMaker JumpStart SDK:

```bash
python workflow_jumpstart_sdk_deploy.py
```

**Demonstrates:**
- Browsing JumpStart model catalog
- One-line model deployment
- Pre-configured optimizations
- Simplified workflow for standard models

**Use cases:**
- Rapid prototyping with popular models
- Standardized deployments across teams
- Benchmarking different model architectures

## Why These Patterns Matter

### LangGraph Integration
- **Statefulness**: Maintain conversation history and context
- **Modularity**: Break complex workflows into manageable steps
- **Debugging**: Inspect and modify execution paths

### LangChain Integration
- **Abstractions**: Unified interface for multiple LLM providers
- **Chaining**: Compose complex operations (retrieval + generation)
- **Ecosystem**: Access to 100+ integrations (vector DBs, tools, etc.)

### JumpStart SDK
- **Speed**: Deploy in 2-3 lines of code
- **Reliability**: Tested configurations from AWS
- **Discovery**: Explore 300+ pre-trained models

## Running Examples

```bash
# Ensure environment is configured
export SAGEMAKER_ROLE_ARN=arn:aws:iam::ACCOUNT:role/ROLE
export HF_TOKEN=your_token  # If needed
export AWS_PROFILE=your-profile  # If needed

# Run any example
cd examples
python agent_stateful_chat_langgraph.py
```

## Building Your Own

Use these examples as templates for:
- RAG (Retrieval-Augmented Generation) systems
- Agent-based workflows
- Multi-model ensembles
- Custom inference pipelines
