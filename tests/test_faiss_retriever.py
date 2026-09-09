from src.ingestion.pdf_loader import load_pdf
from src.ingestion.smart_chunker import smart_chunk_pages
from src.retrieval.embedder import Embedder
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

retriever = FAISSRetriever(
    chunks=chunks,
    embeddings=embeddings,
    embedder=embedder
)

query = "Why is chunking important in RAG?"

results = retriever.retrieve(
    query,
    top_k=5
)

print("=" * 60)
print("FAISS RETRIEVAL TEST")
print("=" * 60)

for rank, result in enumerate(
    results,
    start=1
):
    print(
        f"Rank {rank} | "
        f"Page {result['page']} | "
        f"Score {result['score']:.4f}"
    )

retriever.save_index(
    "data/processed/test_faiss.index"
)

print("\nFAISS index saved successfully.")