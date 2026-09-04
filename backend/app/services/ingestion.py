import os
from pathlib import Path
from typing import List, Dict, Any
from backend.app.config import settings
from backend.app.core.chunker import StructureAwareMarkdownChunker, MarkdownChunk
from backend.app.core.vector_store import ChromaVectorStore
from backend.app.core.embeddings import EmbeddingFactory
from backend.app.models.schemas import IngestResponse, SourcesResponse, DocumentSourceItem

class IngestionService:
    def __init__(self, knowledge_dir: Path = None, vector_store: ChromaVectorStore = None):
        self.knowledge_dir = knowledge_dir or settings.KNOWLEDGE_DIR
        self.vector_store = vector_store or ChromaVectorStore()
        self.chunker = StructureAwareMarkdownChunker(
            max_chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def ingest_all(self, embedder_provider_name: str = None) -> IngestResponse:
        os.makedirs(self.knowledge_dir, exist_ok=True)
        md_files = sorted(list(self.knowledge_dir.glob("*.md")))
        
        all_chunks: List[MarkdownChunk] = []
        categories_found = set()

        for file_path in md_files:
            chunks = self.chunker.chunk_file(file_path)
            all_chunks.extend(chunks)
            if chunks:
                categories_found.add(chunks[0].category)

        # Clear existing collection and upsert fresh chunks
        self.vector_store.clear()
        
        embedder = EmbeddingFactory.get_provider(embedder_provider_name)
        total_upserted = self.vector_store.upsert_chunks(all_chunks, embedder)

        return IngestResponse(
            status="success",
            total_files_processed=len(md_files),
            total_chunks_created=total_upserted,
            indexed_collection=self.vector_store.collection_name,
            categories_found=sorted(list(categories_found))
        )

    def get_sources_overview(self) -> SourcesResponse:
        md_files = sorted(list(self.knowledge_dir.glob("*.md")))
        items: List[DocumentSourceItem] = []
        total_chunks = 0

        for file_path in md_files:
            chunks = self.chunker.chunk_file(file_path)
            total_chunks += len(chunks)
            sections = list(dict.fromkeys([c.section_title for c in chunks]))
            category = file_path.stem.lower()

            items.append(
                DocumentSourceItem(
                    file_name=file_path.name,
                    category=category,
                    total_chunks=len(chunks),
                    sections=sections
                )
            )

        return SourcesResponse(
            total_files=len(md_files),
            total_chunks=total_chunks,
            sources=items
        )
