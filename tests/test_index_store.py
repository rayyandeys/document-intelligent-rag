from src.ingestion.pdf_loader import load_pdf
from src.ingestion.smart_chunker import smart_chunk_pages
from src.retrieval.embedder import Embedder
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.index_store import (
    save_retrieval_index,
    load_retrieval_index,
)


# --------------------------------------------------
# BUILD INDEX
# --------------------------------------------------

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


# --------------------------------------------------
# SAVE INDEX + METADATA
# --------------------------------------------------

save_retrieval_index(
    retriever=retriever,
    chunks=chunks,
    directory="data/processed/sample_index"
)

print("Index and metadata saved.")


# --------------------------------------------------
# LOAD FROM DISK
# --------------------------------------------------

loaded_retriever = load_retrieval_index(
    directory="data/processed/sample_index",
    embedder=embedder
)

print("Index and metadata loaded.")


# --------------------------------------------------
# QUERY LOADED INDEX
# --------------------------------------------------

results = loaded_retriever.retrieve(
    "Why is chunking important in RAG?",
    top_k=5
)


print("\n" + "=" * 60)
print("PERSISTENT INDEX STORE TEST")
print("=" * 60)

print(
    f"Vectors loaded: "
    f"{loaded_retriever.index.ntotal}"
)

print(
    f"Metadata chunks loaded: "
    f"{len(loaded_retriever.chunks)}"
)

for rank, result in enumerate(
    results,
    start=1
):
    print(
        f"Rank {rank} | "
        f"Page {result['page']} | "
        f"Score {result['score']:.4f}"
    )

print(
    "\nPersistent retrieval pipeline "
    "works successfully."
)