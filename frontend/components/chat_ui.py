import streamlit as st
from typing import List, Dict, Any

SUGGESTED_PROMPTS = [
    "📈 Tell me about the Customer LTV & Retention Analytics project",
    "🛡️ How does the Credit Card Fraud Detection model work?",
    "🎓 Explain the Student Performance Indicator architecture",
    "💻 What are Prakhar's core ML, Python & GenAI skills?",
    "🏛️ Tell me about Prakhar's education at IIITDM Jabalpur",
    "📬 How can I contact Prakhar for opportunities?"
]

def render_hero_header():
    st.markdown("""
    <div class="portfolio-hero">
        <h1 class="portfolio-hero-title">Prakhar — AI & ML Portfolio Assistant</h1>
        <div class="portfolio-hero-subtitle">
            Ask me anything about Prakhar's end-to-end Machine Learning pipelines, Customer Analytics & LTV modeling, Credit Card Fraud detection, technical skills, or education at IIITDM Jabalpur. Powered by structure-aware Markdown RAG & ChromaDB.
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_suggested_prompts():
    st.markdown("<p style='font-size: 13px; font-weight: 600; color: #a5b4fc; margin-bottom: 8px;'>💡 Suggested Questions:</p>", unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, prompt in enumerate(SUGGESTED_PROMPTS):
        col = cols[idx % 3]
        clean_text = prompt.split(" ", 1)[1] if " " in prompt else prompt
        if col.button(prompt, key=f"sugg_{idx}", use_container_width=True):
            st.session_state["pending_prompt"] = clean_text
            st.rerun()

def render_citations(sources: List[Dict[str, Any]]):
    if not sources:
        return

    with st.expander(f"🔍 Verified Knowledge Citations ({len(sources)} source chunks retrieved)", expanded=False):
        for i, src in enumerate(sources, 1):
            score_pct = int(src.get("relevance_score", 0.0) * 100)
            st.markdown(f"""
            <div class="citation-box">
                <div class="citation-header">
                    <span>📄 {src.get('source_file')} &nbsp;•&nbsp; <span class="citation-breadcrumb">{src.get('breadcrumb')}</span></span>
                    <span class="meta-chip">Relevance: {score_pct}%</span>
                </div>
                <div style="color: #cbd5e1; font-size: 12px; margin-top: 6px; white-space: pre-wrap; font-family: var(--font-mono); background: rgba(0,0,0,0.25); padding: 8px; border-radius: 6px;">
{src.get('content')}
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_message(msg: Dict[str, Any]):
    role = msg.get("role", "user")
    content = msg.get("content", "")
    sources = msg.get("sources", [])
    model_used = msg.get("model_used", "")
    latency_ms = msg.get("latency_ms", None)

    if role == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)
            
            # Show metadata chips
            if model_used or latency_ms:
                meta_html = "<div style='display: flex; gap: 8px; margin-top: 10px; margin-bottom: 6px;'>"
                if model_used:
                    meta_html += f"<span class='meta-chip'>Model: {model_used}</span>"
                if latency_ms:
                    meta_html += f"<span class='meta-chip'>⏱️ {latency_ms} ms</span>"
                meta_html += "</div>"
                st.markdown(meta_html, unsafe_allow_html=True)

            # Render citation expander
            if sources:
                render_citations(sources)
