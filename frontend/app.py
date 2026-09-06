import sys
import os
from pathlib import Path
import streamlit as st
import requests

# Add project root to sys.path for direct module import if needed
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from frontend.components.sidebar import render_sidebar
from frontend.components.chat_ui import (
    render_hero_header,
    render_suggested_prompts,
    render_message,
    render_citations
)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")

# Streamlit Page Setup
st.set_page_config(
    page_title="Prakhar — AI & ML Portfolio Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_file = BASE_DIR / "frontend" / "styles" / "custom.css"
if css_file.exists():
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "👋 Hi! I am **Prakhar's AI Portfolio Assistant**.\n\nI can answer questions about Prakhar's **Customer LTV & Retention Analytics**, **Credit Card Fraud Detection**, **Student Performance Indicator**, technical skills in Python, Machine Learning, and GenAI, or his academic background at **IIITDM Jabalpur**. What would you like to explore?",
            "sources": [],
            "model_used": "System Persona",
            "latency_ms": None
        }
    ]

if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None

# Render Sidebar
render_sidebar(api_base_url=API_BASE_URL)

# Main UI
render_hero_header()

# Render Suggested Questions if only initial welcome message
if len(st.session_state["messages"]) <= 1:
    render_suggested_prompts()
    st.markdown("<br>", unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state["messages"]:
    render_message(msg)

# Check if there is a pending prompt from suggestion chip
user_input = None
if st.session_state["pending_prompt"]:
    user_input = st.session_state["pending_prompt"]
    st.session_state["pending_prompt"] = None
else:
    user_input = st.chat_input("Ask about Prakhar's ML projects, LTV analytics, fraud detection, skills, or contact info...")

if user_input:
    # 1. Append User Message
    user_msg = {"role": "user", "content": user_input}
    st.session_state["messages"].append(user_msg)
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # 2. Fetch Response from Backend
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Retrieving portfolio knowledge & synthesizing response..."):
            provider = st.session_state.get("selected_provider", "gemini")
            top_k = st.session_state.get("selected_top_k", 4)
            temp = st.session_state.get("selected_temp", 0.7)

            # Build conversation history for context
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["messages"][:-1]
                if m.get("role") in ["user", "assistant"]
            ]

            payload = {
                "query": user_input,
                "conversation_history": history[-4:],
                "llm_provider": provider,
                "top_k": top_k,
                "temperature": temp
            }

            try:
                res = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=25)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No response generated.")
                    sources = data.get("sources", [])
                    model_used = data.get("model_used", provider)
                    latency_ms = data.get("latency_ms", 0.0)

                    # Display response
                    st.markdown(answer)

                    meta_html = f"<div style='display: flex; gap: 8px; margin-top: 10px; margin-bottom: 6px;'><span class='meta-chip'>Model: {model_used}</span><span class='meta-chip'>⏱️ {latency_ms} ms</span></div>"
                    st.markdown(meta_html, unsafe_allow_html=True)

                    if sources:
                        render_citations(sources)

                    # Save to state
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "model_used": model_used,
                        "latency_ms": latency_ms
                    })
                else:
                    err_msg = f"⚠️ Backend returned status {res.status_code}: {res.text}"
                    st.error(err_msg)
            except Exception as e:
                # Direct in-process fallback if backend is not running standalone
                st.warning(f"Could not reach FastAPI server at `{API_BASE_URL}`. Attempting internal fallback...")
                try:
                    from backend.app.core.rag_pipeline import PortfolioRAGPipeline
                    from backend.app.models.schemas import ChatRequest, ChatMessage
                    
                    pipeline = PortfolioRAGPipeline()
                    chat_req = ChatRequest(
                        query=user_input,
                        conversation_history=[ChatMessage(role=m["role"], content=m["content"]) for m in history],
                        llm_provider="offline",
                        top_k=top_k,
                        temperature=temp
                    )
                    resp = pipeline.generate_chat_response(chat_req)
                    
                    st.markdown(resp.answer)
                    if resp.sources:
                        render_citations([s.model_dump() for s in resp.sources])
                        
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": resp.answer,
                        "sources": [s.model_dump() for s in resp.sources],
                        "model_used": resp.model_used,
                        "latency_ms": resp.latency_ms
                    })
                except Exception as inner_e:
                    st.error(f"Fallback failed: {inner_e}")
