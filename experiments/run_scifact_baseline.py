from src.evaluation.scifact_loader import load_scifact
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever


corpus, queries, qrels = load_scifact()

documents = []

for document_id, document in corpus.items():
    text = f"{document['title']}. {document['text']}"

    documents.append({
        "chunk_id": document_id,
        "document_id": document_id,
        "text": text,
        "page": 0,
        "source": "SciFact"
    })


print("=" * 60)
print("SCIFACT DOCUMENT-LEVEL BASELINE")
print("=" * 60)

print(f"Documents: {len(documents)}")
print(f"Evaluation queries: {len(qrels)}")

print("\nLoading embedding model...")

embedder = Embedder()

print("Embedding corpus...")

embeddings = embedder.embed_chunks(documents)

retriever = Retriever(
    chunks=documents,
    embeddings=embeddings,
    embedder=embedder
)


top_k_values = [1, 3, 5, 10]

hits = {
    k: 0
    for k in top_k_values
}

reciprocal_rank_sum = 0.0

print("Running retrieval evaluation...")

for query_id, relevant_document_ids in qrels.items():

    query = queries[query_id]

    results = retriever.retrieve(
        query,
        top_k=max(top_k_values)
    )

    retrieved_ids = [
        str(result["document_id"])
        for result in results
    ]

    for k in top_k_values:
        if any(
            document_id in relevant_document_ids
            for document_id in retrieved_ids[:k]
        ):
            hits[k] += 1

    for rank, document_id in enumerate(
        retrieved_ids,
        start=1
    ):
        if document_id in relevant_document_ids:
            reciprocal_rank_sum += 1.0 / rank
            break


query_count = len(qrels)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

for k in top_k_values:
    hit_rate = hits[k] / query_count

    print(
        f"Hit@{k}: "
        f"{hit_rate:.4f} "
        f"({hits[k]}/{query_count})"
    )

mrr = reciprocal_rank_sum / query_count

print(f"MRR@10: {mrr:.4f}")