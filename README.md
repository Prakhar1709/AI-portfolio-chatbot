# 🤖 AI Chatbot Portfolio Engine

An intelligent, production-grade AI portfolio chatbot built with **FastAPI**, **Streamlit**, **ChromaDB**, **Structure-Aware Markdown RAG**, and **Modular LLM Adapters** (Google Gemini, OpenAI, Groq, and Smart Offline fallback).

---

## 🏛️ Architecture Overview

The system strictly implements the design decisions specified in the architecture blueprint:

```
                  +----------------------------------------------+
                  |           KNOWLEDGE BASE (Markdown)          |
                  |  profile.md  projects.md  experience.md ...  |
                  +----------------------+-----------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |       STRUCTURE-AWARE RECURSIVE CHUNKER      |
                  |  - Preserves H1/H2/H3 header breadcrumbs     |
                  |  - Attaches category, tags & char metadata   |
                  +----------------------+-----------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |         HOSTED / MODULAR EMBEDDINGS          |
                  |   Gemini text-embedding-004 / OpenAI / Local |
                  +----------------------+-----------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |            CHROMADB VECTOR STORE             |
                  |   Persistent local storage with cosine dist  |
                  +----------------------+-----------------------+
                                         |
                       +-----------------+-----------------+
                       |                                   |
                       v                                   v
        +-----------------------------+     +-----------------------------+
        |  SEMANTIC + METADATA SEARCH |     |       FASTAPI BACKEND       |
        |  - Intent-guided filtering  |     |  - POST /api/chat           |
        |  - Metadata score boosting  |     |  - POST /api/ingest         |
        |  - Source citation tracking |     |  - GET  /api/health         |
        +-----------------------------+     +--------------+--------------+
                                                           |
                                                           v
                                            +-----------------------------+
                                            |      STREAMLIT FRONTEND     |
                                            |  - Glassmorphic Dark UI     |
                                            |  - Dynamic Citation Inspector|
                                            |  - Profile & Skills Widget  |
                                            +-----------------------------+
```

---

## 🚀 Key Features

1. **Structured Section Markdown Knowledge**:
   - Ingests markdown files from `data/knowledge/` (`profile.md`, `projects.md`, `experience.md`, `skills.md`, `education.md`, `faq.md`).
2. **Structure-Aware Hierarchical Chunking**:
   - Parses header trees, preserves context breadcrumbs (e.g. `Featured Projects > OmniRAG Enterprise > Architecture & Technical Highlights`), and attaches rich metadata.
3. **ChromaDB Vector Store with Metadata Filtering**:
   - Persistent vector storage with collection management and dynamic re-indexing.
4. **Hybrid Semantic + Metadata Retrieval**:
   - Detects user intent (e.g., project questions vs. skill questions vs. contact inquiries) and applies targeted metadata filters and ranking boosts.
5. **Pluggable LLM Providers**:
   - Supports **Google Gemini** (`gemini-1.5-flash`), **OpenAI** (`gpt-4o-mini`), **Groq** (`llama-3.3-70b-versatile`), and an intelligent **Smart Offline Persona Engine** that synthesizes structured answers without requiring an active API key.
6. **State-of-the-Art Streamlit Interface**:
   - Glassmorphic dark theme, instant suggestion chips, collapsible verified citations with relevance meters, and live model selector.

---

## 📦 Directory Structure

```
ai-portfolio-chatbot/
├── data/
│   ├── knowledge/              # Markdown portfolio knowledge files
│   │   ├── profile.md          # Personal bio, summary, contact
│   │   ├── projects.md         # Detailed showcase of featured projects
│   │   ├── experience.md       # Work history, roles & metrics
│   │   ├── skills.md           # Categorized technical competencies
│   │   ├── education.md        # Education & certifications
│   │   └── faq.md              # Common interview & recruiter questions
│   └── chroma_db/              # Persistent ChromaDB vector store
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI REST endpoints
│   │   ├── config.py           # Configuration & environment loader
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py      # Pydantic models & validation
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── chunker.py      # Structure-aware recursive markdown chunker
│   │   │   ├── embeddings.py   # Modular embedding adapters
│   │   │   ├── vector_store.py # ChromaDB manager
│   │   │   ├── retriever.py    # Semantic + metadata retrieval engine
│   │   │   ├── llm.py          # Modular LLM client adapters
│   │   │   └── rag_pipeline.py # Persona prompt orchestration & citation builder
│   │   └── services/
│   │       ├── __init__.py
│   │       └── ingestion.py    # Auto-indexer for markdown documents
│   └── requirements.txt
├── frontend/
│   ├── app.py                  # Streamlit application entrypoint
│   ├── components/
│   │   ├── chat_ui.py          # Chat bubbles, chips & citation inspector
│   │   └── sidebar.py          # Profile hero, status & settings
│   ├── styles/
│   │   └── custom.css          # Glassmorphic dark theme stylesheet
│   └── requirements.txt
├── .env.example
├── run_app.py                  # Single-command launcher
└── README.md
```

---

## 🛠️ Quick Start

### 1. Configure Environment (Optional)
Copy `.env.example` to `.env` and configure your preferred provider:
```bash
cp .env.example .env
```
*(Note: If no API keys are provided, the app automatically runs in **Smart Offline Persona Mode** with deterministic embeddings and local inference!)*

### 2. Run Both Services Together
```bash
python run_app.py
```
- **Streamlit Frontend**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Backend**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## ⚙️ Customizing Your Portfolio
To personalize the knowledge base with your own background:
1. Edit the markdown files inside `data/knowledge/` with your own projects, experience, and contact links.
2. In the Streamlit sidebar, click **"🔄 Re-index Markdown Files"** or call `POST /api/ingest`.
3. The chatbot will instantly reflect your updated background with verified citations!
