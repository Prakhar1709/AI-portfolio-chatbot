import time
from typing import List, Optional
from backend.app.config import settings
from backend.app.core.vector_store import ChromaVectorStore
from backend.app.core.embeddings import EmbeddingFactory
from backend.app.core.retriever import PortfolioRetriever
from backend.app.core.llm import LLMFactory
from backend.app.models.schemas import ChatRequest, ChatResponse, SourceCitation

SYSTEM_PERSONA_PROMPT = """You are the official AI Portfolio Assistant & Digital Representative for {persona_name}, {persona_title}.

Your mission is to represent {persona_name} accurately, professionally, and enthusiastically to recruiters, engineering managers, clients, and technical peers.

### Core Guidelines:
1. **Fact Grounding**: Base your answers strictly on the retrieved portfolio context provided below. Do not fabricate projects, metrics, or credentials not in the knowledge base.
2. **Tone & Style**: Friendly, professional, articulate, and engineering-focused. Highlight concrete metrics (e.g., latency reductions, scale, cost savings, accuracy scores) whenever discussing projects or experience.
3. **Structure**: Use markdown formatting with clear headings, bullet points, and bold keywords to make your responses easy to scan.
4. **Actionable Links**: When relevant, invite the user to check out demo links, GitHub repositories, or schedule a 30-minute intro call.
5. **Transparency**: If asked something outside {persona_name}'s experience or not found in the portfolio, politely acknowledge it and steer the conversation back to {persona_name}'s core competencies.

<RETRIEVED_PORTFOLIO_CONTEXT>
{context}
</RETRIEVED_PORTFOLIO_CONTEXT>
"""

class PortfolioRAGPipeline:
    def __init__(self):
        self.vector_store = ChromaVectorStore()
        self.embedder = EmbeddingFactory.get_provider()
        self.retriever = PortfolioRetriever(self.vector_store, self.embedder)

    def reload_embedder(self, provider_name: Optional[str] = None):
        self.embedder = EmbeddingFactory.get_provider(provider_name)
        self.retriever = PortfolioRetriever(self.vector_store, self.embedder)

    def generate_chat_response(self, request: ChatRequest) -> ChatResponse:
        start_time = time.time()

        # 1. Retrieve relevant chunks
        top_k = request.top_k or settings.TOP_K_RETRIEVAL
        citations: List[SourceCitation] = self.retriever.retrieve(
            query=request.query,
            top_k=top_k
        )

        # 2. Build context string
        context_blocks = []
        for i, c in enumerate(citations, 1):
            context_blocks.append(
                f"--- Source [{i}] | File: {c.source_file} | Section: {c.breadcrumb} (Relevance: {c.relevance_score}) ---\n{c.content}"
            )
        context_str = "\n\n".join(context_blocks) if context_blocks else "No specific documents found."

        # 3. Assemble System Prompt
        system_prompt = SYSTEM_PERSONA_PROMPT.format(
            persona_name=settings.PERSONA_NAME,
            persona_title=settings.PERSONA_TITLE,
            context=context_str
        )

        # 4. Get LLM Client & Generate Response
        llm_client = LLMFactory.get_client(request.llm_provider)
        llm_result = llm_client.generate_response(
            system_prompt=system_prompt,
            user_prompt=request.query,
            conversation_history=request.conversation_history,
            temperature=request.temperature or 0.7
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return ChatResponse(
            answer=llm_result["text"],
            sources=citations,
            provider_used=llm_result["provider"],
            model_used=llm_result["model"],
            latency_ms=latency_ms,
            retrieved_chunks_count=len(citations)
        )
