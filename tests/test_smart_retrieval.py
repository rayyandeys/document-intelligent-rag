from src.ingestion.pdf_loader import load_pdf
from src.ingestion.smart_chunker import smart_chunk_pages
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.evaluation.retrieval_eval import hit_at_k, reciprocal_rank


pages = load_pdf("data/raw/sample.pdf")

chunks = smart_chunk_pages(
    pages,
    chunk_size=800,
    overlap_sentences=1
)

embedder = Embedder()

embeddings = embedder.embed_chunks(chunks)

retriever = Retriever(
    chunks=chunks,
    embeddings=embeddings,
    embedder=embedder
)

query = "Why is chunking necessary in RAG?"

results = retriever.retrieve(query, top_k=5)

expected_pages = [2]

hit = hit_at_k(results, expected_pages)
rr = reciprocal_rank(results, expected_pages)

print(f"Chunks created: {len(chunks)}")
print(f"Query: {query}")
print(f"Hit@5: {hit}")
print(f"Reciprocal Rank: {rr:.4f}")

print("\nRetrieved pages:")

for rank, result in enumerate(results, start=1):
    print(
        f"Rank {rank}: "
        f"page={result['page']}, "
        f"score={result['score']:.4f}"
    )