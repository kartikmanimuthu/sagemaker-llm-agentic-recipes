# %% [markdown]
# # SageMaker Chat Integration with LangGraph
# 
# This notebook demonstrates how to invoke a DeepSeek SageMaker endpoint using LangGraph for stateful chat management.

# %%
# Install dependencies
# pip install -r requirements.txt -q
# pip install 'sagemaker==2.251.1' --no-deps

import sagemaker
import boto3
import json
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END

# Configuration
# Configuration
PROFILE_NAME = os.environ.get('AWS_PROFILE', 'default')
REGION_NAME = 'us-east-1' # Change to 'ap-south-1' or other regions as needed
ENDPOINT_NAME = "jumpstart-dft-deepseek-llm-r1-disti-20251206-121042"

print(f"Using Endpoint: {ENDPOINT_NAME}")
print(f"Region: {REGION_NAME}")

# %%
# Initialize Predictor
# Create a Boto3 session with the specific profile
try:
    boto_session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)

    print(f"Authenticated with profile: {boto_session.profile_name}")
    print(f"Region: {boto_session.region_name}")

    # Create a SageMaker session using the boto3 session
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    print(f"Authenticated with profile: {boto_session.profile_name}")
except Exception as e:
    print(f"Failed to use profile {PROFILE_NAME}, falling back to default. Error: {e}")
    sagemaker_session = None

# retrieve_default can hang if it tries to download model artifacts. 
# Using direct Predictor initialization is faster and more robust for deployed endpoints.
print("Initializing Predictor...")
try:
    if sagemaker_session:
        predictor = Predictor(
            endpoint_name=ENDPOINT_NAME,
            sagemaker_session=sagemaker_session,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
    else:
        predictor = Predictor(
            endpoint_name=ENDPOINT_NAME,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
    print("Predictor initialized successfully.")
except Exception as e:
    print(f"Error initializing predictor: {e}")
    raise e

# %%
# Define Graph State
class State(TypedDict):
    # Messages list stores the conversation history
    messages: List[Dict[str, str]]

# Define the Chat Node
def call_model(state: State):
    print("Invoking model...")
    messages = state["messages"]
    
    # Prepare payload for DeepSeek model (Chat API format)
    # The endpoint appears to support OpenAI-compatible chat completion format
    payload = {
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        # Invoke endpoint
        # print(f"DEBUG: Sending payload with {len(messages)} messages")
        response = predictor.predict(payload)
        
        # Parse response
        if isinstance(response, bytes):
            response_data = json.loads(response.decode('utf-8'))
        else:
            response_data = response
            
        print(f"DEBUG: Response keys: {response_data.keys() if isinstance(response_data, dict) else 'List'}")
        
        # Extract content
        content = None
        
        # 1. Handle OpenAI-compatible Chat Completion response (Primary target)
        if isinstance(response_data, dict) and 'choices' in response_data:
            choice = response_data['choices'][0]
            if 'message' in choice:
                message = choice['message']
                content = message.get('content')
                
                # Check for reasoning content (DeepSeek R1 feature)
                reasoning = message.get('reasoning_content')
                if reasoning:
                    print(f"DEBUG: Reasoning content found ({len(reasoning)} chars)")
                    
                    # If content is None (e.g. hit max_tokens during reasoning), usage reasoning as valid output
                    if not content:
                        print("DEBUG: Content is empty, using reasoning content as fallback.")
                        content = f"**Reasoning (truncated):**\n{reasoning}\n\n[Response interrupted due to length limit]"
                    else:
                        # Optionally include reasoning in the final output or log it
                        # For now, we'll store specific reasoning if needed, but here we just return the content
                        pass

        # 2. Handle generic list response [{'generated_text': '...'}]
        elif isinstance(response_data, list) and len(response_data) > 0:
            item = response_data[0]
            if isinstance(item, dict):
                 content = item.get('generated_text')
        
        # 3. Handle raw dictionary with 'generated_text'
        elif isinstance(response_data, dict) and 'generated_text' in response_data:
             content = response_data['generated_text']

        if content is None:
             print(f"DEBUG: Failed to extract content. Raw data: {response_data}")
             content = "Error: No content generated."
             
        # Create assistant message
        assistant_message = {"role": "assistant", "content": content}
        
        # Return updated state
        return {"messages": messages + [assistant_message]}
        
    except Exception as e:
        print(f"Error invoking endpoint: {e}")
        return {"messages": messages + [{"role": "assistant", "content": f"Error: {str(e)}"}]}

# Build the Graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("deepseek_agent", call_model)

# Add edges
workflow.add_edge(START, "deepseek_agent")
workflow.add_edge("deepseek_agent", END)

# Compile the graph
app = workflow.compile()

# %%
# Test the Graph with a single turn
initial_state = {
    "messages": [
        {"role": "user", "content": "what is aws sagemaker."}
    ]
}

output = app.invoke(initial_state)
print("\n--- Final Output ---")
print(output)
print(output["messages"][-1]["content"])

# %%
# Interactive Chat Function
def chat_session():
    print("Starting chat session. Type 'quit' to exit.")
    conversation_history = []
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            break
            
        conversation_history.append({"role": "user", "content": user_input})
        
        # Run graph
        result = app.invoke({"messages": conversation_history})
        
        # Update history with the result
        conversation_history = result["messages"]
        
        print(f"Assistant: {conversation_history[-1]['content']}")

# chat_session() # Uncomment to run

# %%
