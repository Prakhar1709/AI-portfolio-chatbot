import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from backend.app.config import settings
from backend.app.core.chunker import MarkdownChunk
from backend.app.core.embeddings import BaseEmbeddingProvider
from backend.app.models.schemas import SourceCitation

class ChromaVectorStore:
    def __init__(self, persist_dir: Path = None, collection_name: str = None):
        self.persist_dir = str(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

    def get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Prakhar AI Portfolio Markdown Knowledge Base"}
        )

    def upsert_chunks(self, chunks: List[MarkdownChunk], embedder: BaseEmbeddingProvider) -> int:
        if not chunks:
            return 0

        col = self.get_collection()
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "category": c.category,
                "section_title": c.section_title,
                "breadcrumb": c.breadcrumb,
                "tags": ",".join(c.tags),
                "char_count": c.char_count
            }
            for c in chunks
        ]

        # Compute embeddings
        embeddings = embedder.embed_documents(documents)

        # Upsert into ChromaDB
        col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        return len(chunks)

    def query_similar(
        self,
        query: str,
        embedder: BaseEmbeddingProvider,
        top_k: int = 4,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[SourceCitation]:
        col = self.get_collection()
        count = col.count()
        if count == 0:
            return []

        query_embedding = embedder.embed_query(query)
        
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, count),
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            query_kwargs["where"] = where_filter

        results = col.query(**query_kwargs)
        
        citations: List[SourceCitation] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return citations

        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            doc_content = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
            
            similarity = max(0.0, 1.0 - (distance / 2.0))

            citations.append(
                SourceCitation(
                    chunk_id=chunk_id,
                    source_file=meta.get("source_file", "unknown.md"),
                    section_title=meta.get("section_title", "General"),
                    breadcrumb=meta.get("breadcrumb", "Knowledge Base"),
                    category=meta.get("category", "general"),
                    content=doc_content,
                    relevance_score=round(similarity, 4),
                    metadata=meta
                )
            )

        return citations

    def get_stats(self) -> Dict[str, Any]:
        col = self.get_collection()
        count = col.count()
        return {
            "collection_name": self.collection_name,
            "total_vectors": count,
            "persist_dir": self.persist_dir
        }

    def clear(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
