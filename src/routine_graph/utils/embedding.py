"""Qwen embeddings from an OpenAI-compatible endpoint; no OpenAI token-ID preprocessing is used."""

import hashlib
import json
import os

import httpx
import numpy as np

from routine_graph.models import normalized_vectors

QUERY_INSTRUCTION = "Retrieve daily activities, habits and routines relevant to the question."


class ActivityEmbedder:
    def __init__(self, model: str | None = None, provider: str | None = None, *, base_url: str | None = None,
                 api_key: str | None = None, dimensions: int | None = None, batch_size: int = 32,
                 timeout: float = 120, client: httpx.Client | None = None, **kwargs):
        self.model_name = model or os.getenv("EMBEDDING_MODEL", "qwen3-embedding-8b")
        self.provider = provider or "openai-compatible"
        self.base_url = (base_url or os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")
        self.api_key = api_key
        self.dimensions = dimensions
        if batch_size < 1 or (dimensions is not None and dimensions < 1):
            raise ValueError("Batch size and embedding dimensions must be positive")
        self.batch_size = batch_size
        self.client = client or httpx.Client(timeout=timeout, transport=httpx.HTTPTransport(retries=2))
        self._owns_client = client is None
        self.model = None
        if self.provider != "openai-compatible":
            from langchain.embeddings import init_embeddings
            self.model = init_embeddings(self.model_name, provider=self.provider, **kwargs)

    @property
    def configuration(self) -> dict:
        return {"model": self.model_name, "provider": self.provider,
                "base_url": self.base_url if self.provider == "openai-compatible" else None, "dimensions": self.dimensions}

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.configuration, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def activity_text(label: str, description: str) -> str:
        return f"{label}: {description}"

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        if self.model is not None:
            return list(normalized_vectors(self.model.embed_documents(texts), self.dimensions))
        key = self.api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set EMBEDDING_API_KEY (or OPENAI_API_KEY) to your embedding API key")
        if not self.base_url:
            raise ValueError("Set EMBEDDING_BASE_URL (or OPENAI_BASE_URL) to your OpenAI-compatible embedding endpoint")
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            payload = {"model": self.model_name, "input": batch, "encoding_format": "float"}
            if self.dimensions is not None:
                payload["dimensions"] = self.dimensions
            response = self.client.post(f"{self.base_url}/embeddings", headers={"Authorization": f"Bearer {key}"}, json=payload)
            response.raise_for_status()
            data = response.json()["data"]
            if sorted(item["index"] for item in data) != list(range(len(batch))):
                raise ValueError("Embedding API returned missing or duplicate input indices")
            vectors.extend(item["embedding"] for item in sorted(data, key=lambda item: item["index"]))
        return list(normalized_vectors(vectors, self.dimensions))

    def embed(self, label: str, description: str) -> np.ndarray:
        return self.embed_texts([self.activity_text(label, description)])[0]

    def embed_query(self, query: str) -> np.ndarray:
        if self.model is not None:
            return normalized_vectors([self.model.embed_query(query)], self.dimensions)[0]
        return self.embed_texts([f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"])[0]

    def close(self):
        if self._owns_client:
            self.client.close()
