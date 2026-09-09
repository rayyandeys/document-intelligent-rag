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

retriever = FAISSRetriever.load_index(
    index_path="data/processed/test_faiss.index",
    chunks=chunks,
    embedder=embedder
)

results = retriever.retrieve(
    "Why is chunking important in RAG?",
    top_k=5
)

print("=" * 60)
print("LOADED FAISS INDEX TEST")
print("=" * 60)

print(f"Vectors loaded: {retriever.index.ntotal}")
print(f"Dimensions: {retriever.dimension}")

for rank, result in enumerate(results, start=1):
    print(
        f"Rank {rank} | "
        f"Page {result['page']} | "
        f"Score {result['score']:.4f}"
    )

print("\nSaved FAISS index loaded successfully.")