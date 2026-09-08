from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever


pages = load_pdf("data/raw/sample.pdf")
chunks = chunk_pages(pages)

embedder = Embedder()

embeddings = embedder.embed_chunks(chunks)

retriever = Retriever(
    chunks=chunks,
    embeddings=embeddings,
    embedder=embedder
)

query = "Why is chunking necessary in RAG?"

results = retriever.retrieve(query, top_k=3)

print(f"\nQuery: {query}")

for rank, result in enumerate(results, start=1):
    print("\n" + "=" * 60)
    print(f"Rank: {rank}")
    print(f"Score: {result['score']:.4f}")
    print(f"Page: {result['page']}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Source: {result['source']}")
    print("=" * 60)
    print(result["text"][:700])