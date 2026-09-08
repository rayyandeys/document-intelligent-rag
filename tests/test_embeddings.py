from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.retrieval.embedder import Embedder


pages = load_pdf("data/raw/sample.pdf")
chunks = chunk_pages(pages)

embedder = Embedder()

embeddings = embedder.embed_chunks(chunks)

print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")
print(f"Embedding matrix shape: {embeddings.shape}")
print(f"Single embedding dimensions: {embeddings.shape[1]}")
print(f"First embedding norm: {round(float((embeddings[0] ** 2).sum() ** 0.5), 4)}")