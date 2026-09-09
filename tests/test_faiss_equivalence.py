import numpy as np

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.smart_chunker import smart_chunk_pages
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.retrieval.faiss_retriever import FAISSRetriever


pages = load_pdf(
    "data/raw/sample.pdf"
)

chunks = smart_chunk_pages(
    pages,
    chunk_size=400,
    overlap_sentences=1
)

embedder = Embedder()

embeddings = embedder.embed_chunks(chunks)

numpy_retriever = Retriever(
    chunks=chunks,
    embeddings=embeddings,
    embedder=embedder
)

faiss_retriever = FAISSRetriever(
    chunks=chunks,
    embeddings=embeddings,
    embedder=embedder
)

query = "Why is chunking important in RAG?"

numpy_results = numpy_retriever.retrieve(
    query,
    top_k=5
)

faiss_results = faiss_retriever.retrieve(
    query,
    top_k=5
)


print("=" * 60)
print("NUMPY vs FAISS EQUIVALENCE TEST")
print("=" * 60)

all_match = True

for rank, (numpy_result, faiss_result) in enumerate(
    zip(numpy_results, faiss_results),
    start=1
):
    same_chunk = (
        numpy_result["chunk_id"]
        == faiss_result["chunk_id"]
    )

    score_close = np.isclose(
        numpy_result["score"],
        faiss_result["score"],
        atol=1e-5
    )

    if not same_chunk or not score_close:
        all_match = False

    print(
        f"Rank {rank} | "
        f"NumPy chunk {numpy_result['chunk_id']} "
        f"({numpy_result['score']:.4f}) | "
        f"FAISS chunk {faiss_result['chunk_id']} "
        f"({faiss_result['score']:.4f}) | "
        f"Match: {same_chunk and score_close}"
    )


print("\nEquivalent retrieval:", all_match)