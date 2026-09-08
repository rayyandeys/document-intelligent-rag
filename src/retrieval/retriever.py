from typing import List, Dict

import numpy as np

from src.retrieval.embedder import Embedder


class Retriever:
    def __init__(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
        embedder: Embedder
    ):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.embedder.embed_query(query)

        scores = np.dot(self.embeddings, query_embedding)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            chunk = self.chunks[index].copy()
            chunk["score"] = float(scores[index])
            results.append(chunk)

        return results