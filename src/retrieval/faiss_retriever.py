from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np

from src.retrieval.embedder import Embedder


class FAISSRetriever:
    def __init__(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
        embedder: Embedder
    ):
        self.chunks = chunks
        self.embedder = embedder

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D matrix."
            )

        self.dimension = embeddings.shape[1]

        # Normalized embeddings + inner product
        # gives cosine similarity.
        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.index.add(embeddings)

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict]:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_embedding = self.embedder.embed_query(
            query
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):
            if index == -1:
                continue

            chunk = self.chunks[index].copy()
            chunk["score"] = float(score)

            results.append(chunk)

        return results

    def save_index(
        self,
        index_path: str
    ) -> None:

        path = Path(index_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(path)
        )

    @classmethod
    def load_index(
        cls,
        index_path: str,
        chunks: List[Dict],
        embedder: Embedder
    ):
        path = Path(index_path)

        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {path}"
            )

        index = faiss.read_index(
            str(path)
        )

        retriever = cls.__new__(cls)

        retriever.chunks = chunks
        retriever.embedder = embedder
        retriever.index = index
        retriever.dimension = index.d

        if index.ntotal != len(chunks):
            raise ValueError(
                "FAISS index size does not match "
                "the number of supplied chunks."
            )

        return retriever