# 🤖 AI Chatbot Portfolio Engine

An intelligent, production-grade AI portfolio chatbot built with **FastAPI**, **Streamlit**, **ChromaDB**, **Structure-Aware Markdown RAG**, and **Modular LLM Adapters**.

The chatbot allows recruiters and visitors to interact with Prakhar's portfolio using natural language and receive context-aware answers with verified source citations.

---

## 🌐 Live Demo

### 🚀 Try the AI Portfolio Chatbot

**[https://prakhar-ai-portfolio.streamlit.app](https://prakhar-ai-portfolio.streamlit.app)**

Ask questions about:

- Projects
- Technical skills
- Experience
- Education
- Tech stack
- Data Science / ML work
- GenAI projects
- Contact information

---

## 🏗️ Architecture Overview

```text
                    ┌─────────────────────────────────────────┐
                    │       KNOWLEDGE BASE (Markdown)         │
                    │ profile.md | projects.md | skills.md   │
                    │ experience.md | education.md | faq.md  │
                    └───────────────────┬─────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │   STRUCTURE-AWARE RECURSIVE CHUNKER     │
                    │ • Preserves H1/H2/H3 breadcrumbs        │
                    │ • Adds category, tags & metadata        │
                    └───────────────────┬─────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │          MODULAR EMBEDDINGS             │
                    │       Local / Gemini / OpenAI           │
                    └───────────────────┬─────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │            CHROMADB VECTOR STORE        │
                    │       Persistent vector storage         │
                    └───────────────────┬─────────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
             ┌──────────────────────┐     ┌────────────────────────┐
             │ SEMANTIC + METADATA  │     │    FASTAPI BACKEND     │
             │      RETRIEVAL       │     │                        │
             │ • Intent filtering   │     │ • /api/chat            │
             │ • Metadata boosting  │     │ • /api/ingest          │
             │ • Source tracking    │     │ • /api/health          │
             └──────────┬───────────┘     └────────────┬───────────┘
                        │                              │
                        └──────────────┬───────────────┘
                                       ▼
                    ┌─────────────────────────────────────────┐
                    │         STREAMLIT FRONTEND              │
                    │ • Glassmorphic dark UI                 │
                    │ • Chat interface                        │
                    │ • Citation inspector                    │
                    │ • Profile & skills sidebar              │
                    │ • LLM provider selector                 │
                    └─────────────────────────────────────────┘
```

---

## 🚀 Key Features

### 1. Structured Markdown Knowledge Base

Portfolio information is maintained as modular Markdown files inside:

```text
data/knowledge/
```

Files include:

- `profile.md`
- `projects.md`
- `experience.md`
- `skills.md`
- `education.md`
- `faq.md`

### 2. Structure-Aware Hierarchical Chunking

The RAG pipeline preserves Markdown heading hierarchy and creates contextual breadcrumbs such as:

```text
Featured Projects
    >
OmniRAG Enterprise
    >
Architecture & Technical Highlights
```

This improves retrieval accuracy and keeps each retrieved chunk connected to its original context.

### 3. ChromaDB Vector Store

Uses **ChromaDB** for persistent vector storage and semantic similarity search.

The system supports:

- Persistent collections
- Dynamic re-indexing
- Metadata filtering
- Similarity-based retrieval

### 4. Hybrid Retrieval

The retrieval system combines:

- Semantic similarity
- Metadata filtering
- Intent detection
- Metadata-based ranking boosts
- Source citation tracking

This allows the chatbot to distinguish between questions about projects, skills, experience, education, and contact information.

### 5. Modular LLM Providers

The application supports multiple LLM providers:

| Provider | Model |
|---|---|
| 🚀 Groq | `openai/gpt-oss-20b` |
| ✨ Google Gemini | Configurable |
| ⚡ OpenAI | Configurable |
| 🛡️ Offline | Smart Offline Persona Engine |

### 6. Modern Streamlit Interface

The frontend includes:

- Glassmorphic dark UI
- Chat interface
- Suggestion chips
- Model/provider selector
- Verified source citations
- Citation relevance information
- Profile and skills sidebar
- Backend health status

---

## 🧰 Technology Stack

| Category | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| Language | Python |
| Vector Database | ChromaDB |
| RAG | Structure-Aware Markdown RAG |
| Embeddings | Local / Gemini / OpenAI |
| LLM | Groq GPT-OSS 20B |
| Validation | Pydantic |
| API Server | Uvicorn |
| Deployment | Streamlit Community Cloud + Render |

---

## 📦 Project Structure

```text
ai-portfolio-chatbot/
│
├── data/
│   ├── knowledge/
│   │   ├── profile.md
│   │   ├── projects.md
│   │   ├── experience.md
│   │   ├── skills.md
│   │   ├── education.md
│   │   └── faq.md
│   │
│   └── chroma_db/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── chunker.py
│   │   │   ├── embeddings.py
│   │   │   ├── vector_store.py
│   │   │   ├── retriever.py
│   │   │   ├── llm.py
│   │   │   └── rag_pipeline.py
│   │   │
│   │   └── services/
│   │       ├── __init__.py
│   │       └── ingestion.py
│   │
│   └── requirements.txt
│
├── frontend/
│   ├── app.py
│   │
│   ├── components/
│   │   ├── chat_ui.py
│   │   └── sidebar.py
│   │
│   ├── styles/
│   │   └── custom.css
│   │
│   └── requirements.txt
│
├── .env.example
├── run_app.py
└── README.md
```

---

## 🔌 API Endpoints

### Chat

```http
POST /api/chat
```

Generates a context-aware portfolio response using retrieved knowledge and the selected LLM provider.

### Ingest Knowledge Base

```http
POST /api/ingest
```

Re-indexes the Markdown knowledge base into ChromaDB.

### Health Check

```http
GET /api/health
```

Returns backend health, active providers, and vector database information.

### Sources

```http
GET /api/sources
```

Returns indexed portfolio knowledge sources.

### Portfolio Summary

```http
GET /api/portfolio/summary
```

Returns a structured summary of the portfolio knowledge base.

---

## 🛠️ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Prakhar1709/AI-portfolio-chatbot.git
cd AI-portfolio-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Install Frontend Dependencies

```bash
pip install -r frontend/requirements.txt
```

### 5. Configure Environment

Create a `.env` file using:

```bash
cp .env.example .env
```

Configure the required API keys for your selected LLM provider.

### 6. Run the Application

```bash
python run_app.py
```

The application will be available at:

```text
Streamlit Frontend:
http://localhost:8501

FastAPI Backend:
http://localhost:8000

FastAPI Swagger Docs:
http://localhost:8000/docs
```

---

## 🧠 How the RAG Pipeline Works

```text
Markdown Knowledge Base
          │
          ▼
Structure-Aware Chunking
          │
          ▼
Embedding Generation
          │
          ▼
ChromaDB Vector Store
          │
          ▼
User Query
          │
          ▼
Intent Detection
          │
          ▼
Semantic + Metadata Retrieval
          │
          ▼
Relevant Context
          │
          ▼
LLM Generation
          │
          ▼
Answer + Source Citations
```

The system retrieves the most relevant portfolio information before generating the final response.

---

## 📚 Updating the Portfolio Knowledge

To update the chatbot's knowledge:

1. Edit the Markdown files inside:

```text
data/knowledge/
```

2. Update information about:

```text
Projects
Experience
Skills
Education
Profile
FAQs
```

3. Re-index the knowledge base using the Streamlit interface or:

```http
POST /api/ingest
```

The chatbot will then use the updated information in future responses.

---

## 💬 Example Questions

Try asking the deployed chatbot:

```text
What are Prakhar's top skills?
```

```text
What projects has Prakhar worked on?
```

```text
Explain Prakhar's credit card fraud project.
```

```text
What is Prakhar's tech stack?
```

```text
Tell me about Prakhar's experience.
```

```text
What machine learning projects has Prakhar built?
```

```text
What GenAI technologies does Prakhar know?
```

---

## ☁️ Deployment

The application uses a split deployment architecture:

```text
                User
                  │
                  ▼
       ┌─────────────────────┐
       │ Streamlit Cloud     │
       │ Frontend            │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Render              │
       │ FastAPI Backend     │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Groq                │
       │ GPT-OSS 20B         │
       └─────────────────────┘
```

### Current Deployment

- **Frontend:** Streamlit Community Cloud
- **Backend:** Render
- **LLM:** Groq `openai/gpt-oss-20b`
- **Embeddings:** Local
- **Vector Database:** ChromaDB

### 🔗 Live Application

**https://prakhar-ai-portfolio.streamlit.app**

---

## 🔐 Environment Variables

API keys should never be committed to GitHub.

Example:

```env
DEFAULT_LLM_PROVIDER=groq
DEFAULT_EMBEDDING_PROVIDER=local
GROQ_API_KEY=your_groq_api_key
HOST=0.0.0.0
```

Keep sensitive credentials inside `.env` locally or environment variables on the deployment platform.

---

## 👨‍💻 Author

### Prakhar Pratap Singh

**AI/ML Developer | Data Science | GenAI**

GitHub:

**https://github.com/Prakhar1709**

---

## ⭐ Project

If you find this project useful or interesting, consider giving the repository a ⭐.

### 🔗 Repository

**https://github.com/Prakhar1709/AI-portfolio-chatbot**

### 🚀 Live Demo

**https://prakhar-ai-portfolio.streamlit.app**