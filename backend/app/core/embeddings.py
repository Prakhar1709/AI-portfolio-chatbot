import os
import math
import hashlib
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from backend.app.config import settings

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str = None, model: str = "models/text-embedding-004"):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.genai = genai

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(response['embedding'])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_query"
        )
        return response['embedding']

class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str = None, model: str = "text-embedding-3-small"):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding

class FallbackLocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic hashed N-gram embedding provider for local/offline execution without external API keys.
    Generates normalized 384-dimensional dense vectors preserving word & semantic ngram similarities.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed_single(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec.tolist()

        # Unigrams, Bigrams, and sub-words
        tokens = list(words)
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
            
        for token in tokens:
            h = int(hashlib.sha256(token.encode('utf-8')).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            weight = math.log(1.0 + len(token))
            vec[idx] += sign * weight

        # L2 Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)

class EmbeddingFactory:
    @staticmethod
    def get_provider(provider_name: str = None) -> BaseEmbeddingProvider:
        provider = (provider_name or settings.DEFAULT_EMBEDDING_PROVIDER).lower()

        if provider == "gemini" and settings.GEMINI_API_KEY:
            try:
                return GeminiEmbeddingProvider()
            except Exception as e:
                print(f"[Embeddings] Gemini init failed ({e}), falling back to local embedder.")

        if provider == "openai" and settings.OPENAI_API_KEY:
            try:
                return OpenAIEmbeddingProvider()
            except Exception as e:
                print(f"[Embeddings] OpenAI init failed ({e}), falling back to local embedder.")

        return FallbackLocalEmbeddingProvider()
