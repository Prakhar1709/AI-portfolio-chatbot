import streamlit as st
import requests
from typing import Dict, Any

def render_sidebar(api_base_url: str):
    with st.sidebar:
        # 1. Profile Avatar & Card
        st.markdown("""
        <div class="sidebar-profile-card">
            <div class="sidebar-avatar">👨‍💻</div>
            <h2 class="sidebar-name">Prakhar</h2>
            <div class="sidebar-title">AI/ML Developer & GenAI Engineer</div>
            <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">📍 Faridabad, India • IIITDM Jabalpur '27</div>
            <div style="margin-top: 10px;">
                <span class="status-pill">
                    <span class="status-dot"></span> Open to Opportunities
                </span>
            </div>
            <div class="badge-group">
                <span class="tech-badge">Python</span>
                <span class="tech-badge">Machine Learning</span>
                <span class="tech-badge">XGBoost</span>
                <span class="tech-badge">LangChain</span>
                <span class="tech-badge">RAG</span>
                <span class="tech-badge">FastAPI</span>
                <span class="tech-badge">Streamlit</span>
                <span class="tech-badge">SQL</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Direct Social / Contact Links
        st.markdown("### 🌐 Connect & Links")
        col1, col2 = st.columns(2)
        with col1:
            st.link_button("💼 LinkedIn", "https://www.linkedin.com/in/prakhar050/", use_container_width=True)
            st.link_button("🐙 GitHub", "https://github.com/Prakhar1709", use_container_width=True)
        with col2:
            st.link_button("✉️ Email", "mailto:workwithprakhar17@gmail.com", use_container_width=True)
            st.link_button("📂 Repositories", "https://github.com/Prakhar1709?tab=repositories", use_container_width=True)

        st.divider()

        # 3. Model & Retrieval Settings
        st.markdown("### ⚙️ Engine Settings")
        
        provider = st.selectbox(
            "LLM Provider",
            options=["gemini", "openai", "groq", "offline"],
            format_func=lambda x: {
                "gemini": "✨ Google Gemini (1.5 Flash)",
                "openai": "⚡ OpenAI (GPT-4o Mini)",
                "groq": "🚀 Groq (GPT-OSS 20B)",
                "offline": "🛡️ Smart Offline Persona Engine"
            }.get(x, x),
            index=2
        )
        st.session_state["selected_provider"] = provider

        col_k, col_t = st.columns(2)
        with col_k:
            top_k = st.slider("Top-K Citations", min_value=2, max_value=8, value=4, step=1)
            st.session_state["selected_top_k"] = top_k
        with col_t:
            temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
            st.session_state["selected_temp"] = temp

        st.divider()

        # 4. Vector Database & Ingestion Control
        st.markdown("### 📚 Knowledge Base Index")
        
        # Check backend health
        backend_online = False
        vector_count = 0
        try:
            res = requests.get(f"{api_base_url}/health", timeout=2)
            if res.status_code == 200:
                backend_online = True
                data = res.json()
                vector_count = data.get("total_vectors_in_db", 0)
        except Exception:
            backend_online = False

        if backend_online:
            st.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 10px; margin-bottom: 12px;">
                <div style="font-size: 12px; color: #34d399; font-weight: 600;">⚡ Backend: Online</div>
                <div style="font-size: 13px; color: #e2e8f0; margin-top: 4px;">Indexed Chunks: <b>{vector_count} vectors</b></div>
                <div style="font-size: 11px; color: #9ca3af;">Storage: ChromaDB (Persistent)</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ FastAPI Backend offline or starting...")

        if st.button("🔄 Re-index Markdown Files", use_container_width=True):
            with st.spinner("Parsing markdown & updating ChromaDB embeddings..."):
                try:
                    res = requests.post(f"{api_base_url}/ingest", timeout=30)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"✅ Indexed {data.get('total_chunks_created', 0)} chunks across {data.get('total_files_processed', 0)} markdown files!")
                        st.rerun()
                    else:
                        st.error(f"Failed to ingest: {res.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")

        # Clear Chat Button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
