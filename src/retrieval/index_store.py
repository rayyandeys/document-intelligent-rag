import json
from pathlib import Path
from typing import List, Dict

from src.retrieval.embedder import Embedder
from src.retrieval.faiss_retriever import FAISSRetriever


def save_retrieval_index(
    retriever: FAISSRetriever,
    chunks: List[Dict],
    directory: str
) -> None:

    path = Path(directory)

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    index_path = path / "vectors.index"
    metadata_path = path / "chunks.json"

    retriever.save_index(
        str(index_path)
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


def load_retrieval_index(
    directory: str,
    embedder: Embedder
) -> FAISSRetriever:

    path = Path(directory)

    index_path = path / "vectors.index"
    metadata_path = path / "chunks.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Chunk metadata not found: {metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    return FAISSRetriever.load_index(
        index_path=str(index_path),
        chunks=chunks,
        embedder=embedder
    )