import streamlit as st
import os
import json
from groq import Groq
from agent import fetch_dataset, train_random_forest, evaluate_model

# Initialize API client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.set_page_config(page_title="Genesis AI", layout="wide")

# 1. Inject the custom Genesis CSS
with open("genesis interface.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 2. Build the full dashboard HTML structure
st.markdown("""
<div class="genesis-app">
<aside class="genesis-sidebar">
<div class="genesis-brand">
<div class="genesis-brand-mark">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
</div>
Genesis
</div>
<div class="genesis-nav-label">WORKSPACE</div>
<nav class="genesis-nav">
<a href="#" class="genesis-nav-link is-active"><span>Overview</span></a>
<a href="#" class="genesis-nav-link"><span>Live runs</span></a>
<a href="#" class="genesis-nav-link"><span>Memory</span></a>
</nav>
<div class="genesis-sidebar-footer">
<div class="genesis-user">
<div class="genesis-avatar">AY</div>
<div>
<div class="genesis-user-name">Ansh Yewale</div>
<div class="genesis-user-meta">Administrator</div>
</div>
</div>
</div>
</aside>
<main class="genesis-main">
<div class="genesis-topbar">
<div class="genesis-breadcrumbs">Workspace / <strong>Overview</strong></div>
<div class="genesis-status genesis-status--neutral">System online</div>
</div>
<div class="genesis-content">
<div class="genesis-eyebrow">AUTONOMOUS INTELLIGENCE</div>
<h1 class="genesis-page-heading">Good evening, Ansh.</h1>
<p class="genesis-page-subtitle">Genesis is monitoring your workspace and is ready to turn intent into action.</p>
<div class="genesis-grid genesis-grid--2 genesis-mt-lg">
<div class="genesis-card">
<div class="genesis-metric">
<div class="genesis-metric-label">Tasks completed</div>
<div class="genesis-metric-value">1,284</div>
</div>
</div>
<div class="genesis-card">
<div class="genesis-agent-state">
<div class="genesis-agent-orb">
<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/></svg>
</div>
<div>
<div class="genesis-section-heading">Genesis is thinking</div>
<div class="genesis-muted">Synthesizing signals across your workspace.</div>
</div>
</div>
</div>
</div>
</div>
</main>
</div>
""", unsafe_allow_html=True)

# ... (Keep your tools list, tool_map, and Streamlit chat loop logic exactly as it is below this) ...
st.title("🧠 Genesis: Autonomous ML Agent")

# Define the tools schema (exactly as it was in agent.py)
tools = [
    {
        "type": "function",
        "function": {
            "name": "fetch_dataset",
            "description": "Downloads a CSV dataset from a URL or Hugging Face.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "dataset_name": {"type": "string"}
                },
                "required": ["source", "dataset_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "train_random_forest",
            "description": "Trains a Random Forest classifier on a specific CSV file and target column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string"},
                    "target_column": {"type": "string"}
                },
                "required": ["csv_path", "target_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_model",
            "description": "Evaluates a trained model and generates confusion matrix and feature importance PNG plots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {"type": "string"},
                    "csv_path": {"type": "string"},
                    "target_column": {"type": "string"}
                },
                "required": ["model_path", "csv_path", "target_column"]
            }
        }
    }
]

tool_map = {
    "fetch_dataset": fetch_dataset,
    "train_random_forest": train_random_forest,
    "evaluate_model": evaluate_model
}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are Genesis, an autonomous ML engineering agent."}
    ]

# Display chat history
for message in st.session_state.messages:
    if message["role"] not in ["system", "tool"] and not getattr(message, "tool_calls", None):
        with st.chat_message(message["role"]):
            st.markdown(message.get("content", ""))

# Handle new user input
if prompt := st.chat_input("Type a command for Genesis..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Execute the Groq reasoning loop
    with st.chat_message("assistant"):
        with st.spinner("Genesis is thinking..."):
            while True:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=st.session_state.messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.2
                )
                
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls
                
                if not tool_calls:
                    # No more tools to call, display final answer
                    st.markdown(response_message.content)
                    st.session_state.messages.append({"role": "assistant", "content": response_message.content})
                    break
                
                # Handle tool execution
                st.session_state.messages.append(response_message)
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Show a temporary UI status for tool execution
                    st.info(f"🛠️ Executing: {function_name}...")
                    
                    function_to_call = tool_map[function_name]
                    function_response = function_to_call(**function_args)
                    
                    st.session_state.messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    })