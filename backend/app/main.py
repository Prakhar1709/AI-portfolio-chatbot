from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import settings
from backend.app.models.schemas import (
    ChatRequest,
    ChatResponse,
    IngestResponse,
    SourcesResponse,
    HealthResponse
)
from backend.app.core.rag_pipeline import PortfolioRAGPipeline
from backend.app.services.ingestion import IngestionService

# Global service instances
rag_pipeline: PortfolioRAGPipeline = None
ingestion_service: IngestionService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline, ingestion_service
    print("[FastAPI] Initializing RAG Pipeline and Ingestion Service...")
    rag_pipeline = PortfolioRAGPipeline()
    ingestion_service = IngestionService(vector_store=rag_pipeline.vector_store)
    
    # Auto-index if database is empty or on startup
    print("[FastAPI] Auto-indexing Prakhar's knowledge base...")
    ingestion_service.ingest_all()
    print("[FastAPI] Prakhar's knowledge base indexed successfully!")
    yield
    print("[FastAPI] Shutting down...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="High-Performance Markdown-RAG Backend for Prakhar's AI Portfolio Chatbot",
    lifespan=lifespan
)

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    stats = rag_pipeline.vector_store.get_stats()
    return HealthResponse(
        status="healthy",
        service=settings.PROJECT_NAME,
        version="1.0.0",
        llm_provider=settings.DEFAULT_LLM_PROVIDER,
        embedding_provider=settings.DEFAULT_EMBEDDING_PROVIDER,
        total_vectors_in_db=stats["total_vectors"],
        active_collection=stats["collection_name"]
    )

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        response = rag_pipeline.generate_chat_response(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Chat Error: {str(e)}")

@app.post("/api/ingest", response_model=IngestResponse)
def reindex_knowledge_base():
    try:
        result = ingestion_service.ingest_all()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.get("/api/sources", response_model=SourcesResponse)
def get_sources():
    try:
        return ingestion_service.get_sources_overview()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sources overview failed: {str(e)}")

@app.get("/api/portfolio/summary")
def get_portfolio_summary():
    return {
        "persona_name": settings.PERSONA_NAME,
        "persona_title": settings.PERSONA_TITLE,
        "location": settings.LOCATION,
        "college": settings.COLLEGE,
        "core_domains": [
            "Customer Analytics & LTV Modeling",
            "Credit Card Fraud Detection",
            "End-to-End Machine Learning Pipelines",
            "Generative AI & RAG Architectures"
        ],
        "quick_links": {
            "github": "https://github.com/Prakhar1709",
            "linkedin": "https://www.linkedin.com/in/prakhar050/",
            "email": "workwithprakhar17@gmail.com"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
