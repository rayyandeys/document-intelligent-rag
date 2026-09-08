from src.ingestion.pdf_loader import load_pdf
from src.ingestion.smart_chunker import smart_chunk_pages
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.generation.generator import Generator


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

query = "Why is chunking important in a RAG system?"

retrieved_chunks = retriever.retrieve(
    query,
    top_k=5
)

print("\nRetrieved evidence:")

for result in retrieved_chunks:
    print(
        f"Page {result['page']} | "
        f"Score: {result['score']:.4f}"
    )

generator = Generator()

answer = generator.generate(
    query=query,
    retrieved_chunks=retrieved_chunks
)

print("\nGenerated answer:")
print("=" * 60)
print(answer)
print("=" * 60)