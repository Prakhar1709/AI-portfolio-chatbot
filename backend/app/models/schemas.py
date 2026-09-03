from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user', 'assistant', or 'system'")
    content: str = Field(..., description="The message body")

class SourceCitation(BaseModel):
    chunk_id: str
    source_file: str
    section_title: str
    breadcrumb: str
    category: str
    content: str
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's input query or question")
    conversation_history: List[ChatMessage] = Field(default_factory=list, description="Previous messages for context")
    llm_provider: Optional[str] = Field(None, description="Override LLM provider (gemini, openai, groq, offline)")
    top_k: Optional[int] = Field(4, description="Number of context chunks to retrieve")
    temperature: Optional[float] = Field(0.7, description="Generation temperature (0.0 to 1.0)")

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceCitation] = Field(default_factory=list)
    provider_used: str
    model_used: str
    latency_ms: float
    retrieved_chunks_count: int

class IngestResponse(BaseModel):
    status: str
    total_files_processed: int
    total_chunks_created: int
    indexed_collection: str
    categories_found: List[str]

class DocumentSourceItem(BaseModel):
    file_name: str
    category: str
    total_chunks: int
    sections: List[str]

class SourcesResponse(BaseModel):
    total_files: int
    total_chunks: int
    sources: List[DocumentSourceItem]

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    llm_provider: str
    embedding_provider: str
    total_vectors_in_db: int
    active_collection: str
