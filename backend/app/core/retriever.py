import re
from typing import List, Optional, Dict
from backend.app.core.vector_store import ChromaVectorStore
from backend.app.core.embeddings import BaseEmbeddingProvider
from backend.app.models.schemas import SourceCitation

class PortfolioRetriever:
    """
    Intelligent retriever combining semantic vector search with query intent detection
    and metadata filtering/boosting.
    """
    CATEGORY_KEYWORDS: Dict[str, List[str]] = {
        "projects": [
            "project", "projects", "built", "repo", "github", "ltv", "retention", "customer ltv",
            "fraud", "credit card", "student", "performance indicator", "indicator", "xgboost",
            "catboost", "smote", "rfm", "olist", "cohort", "power bi", "flask", "app", "applications"
        ],
        "skills": [
            "skill", "skills", "stack", "tech", "technologies", "framework", "frameworks",
            "python", "sql", "fastapi", "flask", "streamlit", "xgboost", "catboost", "scikit-learn",
            "langchain", "langgraph", "rag", "chromadb", "machine learning", "ml", "genai", "power bi"
        ],
        "experience": [
            "experience", "career", "job", "role", "company", "work", "history", "drdo", "intern",
            "internship", "research", "contributor", "developer", "open source"
        ],
        "profile": [
            "who is", "about", "bio", "contact", "email", "phone", "reach", "linkedin", "location",
            "faridabad", "resume", "values", "philosophy", "prakhar"
        ],
        "faq": [
            "why hire", "hire you", "interview", "availability", "opportunity", "class imbalance",
            "approach", "hallucination", "rag approach", "methodology"
        ],
        "education": [
            "education", "degree", "university", "college", "iiitdm", "jabalpur", "b.tech",
            "btech", "school", "graduating", "2027", "batch"
        ]
    }

    def __init__(self, vector_store: ChromaVectorStore, embedder: BaseEmbeddingProvider):
        self.vector_store = vector_store
        self.embedder = embedder

    def detect_category_intent(self, query: str) -> Optional[str]:
        q_lower = query.lower()
        scores = {}
        for cat, keywords in self.CATEGORY_KEYWORDS.items():
            count = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', q_lower))
            if count > 0:
                scores[cat] = count

        if scores:
            # Pick highest matched category
            return max(scores, key=scores.get)
        return None

    def retrieve(self, query: str, top_k: int = 4) -> List[SourceCitation]:
        # 1. Global semantic retrieval
        global_results = self.vector_store.query_similar(
            query=query,
            embedder=self.embedder,
            top_k=top_k + 2
        )

        # 2. Metadata-guided category intent retrieval
        detected_category = self.detect_category_intent(query)
        cat_results = []
        if detected_category:
            cat_results = self.vector_store.query_similar(
                query=query,
                embedder=self.embedder,
                top_k=top_k,
                where_filter={"category": detected_category}
            )

        # 3. Merge & Re-rank with metadata boosting
        merged_map = {}
        
        # Add category results with a boost
        for res in cat_results:
            boosted_score = min(1.0, res.relevance_score * 1.15)
            res.relevance_score = round(boosted_score, 4)
            merged_map[res.chunk_id] = res

        # Add global results
        for res in global_results:
            if res.chunk_id not in merged_map:
                merged_map[res.chunk_id] = res
            else:
                # Keep highest score
                if res.relevance_score > merged_map[res.chunk_id].relevance_score:
                    merged_map[res.chunk_id] = res

        # Sort by relevance score descending
        sorted_citations = sorted(merged_map.values(), key=lambda x: x.relevance_score, reverse=True)
        return sorted_citations[:top_k]
