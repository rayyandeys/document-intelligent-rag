from src.evaluation.scifact_loader import load_scifact
from src.ingestion.smart_chunker import smart_chunk_pages
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever


CHUNK_SIZE = 600
OVERLAP_SENTENCES = 1
TOP_K_VALUES = [1, 3, 5, 10]


corpus, queries, qrels = load_scifact()

chunks = []

for document_id, document in corpus.items():

    document_text = f"{document['title']}. {document['text']}"

    pages = [{
        "page": 0,
        "text": document_text,
        "source": document_id
    }]

    document_chunks = smart_chunk_pages(
        pages,
        chunk_size=CHUNK_SIZE,
        overlap_sentences=OVERLAP_SENTENCES
    )

    for chunk in document_chunks:
        chunk["document_id"] = document_id
        chunks.append(chunk)


print("=" * 60)
print("SCIFACT CHUNK-LEVEL RETRIEVAL")
print("=" * 60)

print(f"Documents: {len(corpus)}")
print(f"Chunks: {len(chunks)}")
print(f"Evaluation queries: {len(qrels)}")
print(f"Chunk size: {CHUNK_SIZE}")

print("\nLoading embedding model...")

embedder = Embedder()

print("Embedding chunks...")

embeddings = embedder.embed_chunks(chunks)

retriever = Retriever(
    chunks=chunks,
    embeddings=embeddings,
    embedder=embedder
)


hits = {k: 0 for k in TOP_K_VALUES}
reciprocal_rank_sum = 0.0

print("Running retrieval evaluation...")


for query_id, relevant_document_ids in qrels.items():

    query = queries[query_id]

    # Retrieve more chunks than the final document cutoff because
    # multiple chunks may belong to the same document.
    results = retriever.retrieve(
        query,
        top_k=50
    )

    retrieved_document_ids = []

    for result in results:

        document_id = str(result["document_id"])

        if document_id not in retrieved_document_ids:
            retrieved_document_ids.append(document_id)

        if len(retrieved_document_ids) >= max(TOP_K_VALUES):
            break


    for k in TOP_K_VALUES:

        if any(
            document_id in relevant_document_ids
            for document_id in retrieved_document_ids[:k]
        ):
            hits[k] += 1


    for rank, document_id in enumerate(
        retrieved_document_ids,
        start=1
    ):

        if document_id in relevant_document_ids:
            reciprocal_rank_sum += 1.0 / rank
            break


query_count = len(qrels)


print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

for k in TOP_K_VALUES:

    hit_rate = hits[k] / query_count

    print(
        f"Hit@{k}: "
        f"{hit_rate:.4f} "
        f"({hits[k]}/{query_count})"
    )


mrr = reciprocal_rank_sum / query_count

print(f"MRR@10: {mrr:.4f}")