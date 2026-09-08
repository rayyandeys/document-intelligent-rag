import csv

from src.evaluation.scifact_loader import load_scifact
from src.ingestion.smart_chunker import smart_chunk_pages
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever


CHUNK_SIZES = [400, 600, 800, 1000, 1200]
TOP_K_VALUES = [1, 3, 5, 10]
OVERLAP_SENTENCES = 1


corpus, queries, qrels = load_scifact()

print("=" * 75)
print("SCIFACT CHUNK SIZE EXPERIMENT")
print("=" * 75)

print(f"Documents: {len(corpus)}")
print(f"Evaluation queries: {len(qrels)}")

print("\nLoading embedding model...")

embedder = Embedder()

experiment_results = []


for chunk_size in CHUNK_SIZES:

    print("\n" + "=" * 75)
    print(f"CHUNK SIZE: {chunk_size}")
    print("=" * 75)

    chunks = []

    for document_id, document in corpus.items():

        document_text = (
            f"{document['title']}. "
            f"{document['text']}"
        )

        pages = [{
            "page": 0,
            "text": document_text,
            "source": document_id
        }]

        document_chunks = smart_chunk_pages(
            pages,
            chunk_size=chunk_size,
            overlap_sentences=OVERLAP_SENTENCES
        )

        for chunk in document_chunks:
            chunk["document_id"] = document_id
            chunks.append(chunk)

    print(f"Chunks created: {len(chunks)}")
    print("Embedding chunks...")

    embeddings = embedder.embed_chunks(chunks)

    retriever = Retriever(
        chunks=chunks,
        embeddings=embeddings,
        embedder=embedder
    )

    hits = {k: 0 for k in TOP_K_VALUES}
    reciprocal_rank_sum = 0.0

    print("Evaluating 300 queries...")

    for query_id, relevant_document_ids in qrels.items():

        results = retriever.retrieve(
            queries[query_id],
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

    result = {
        "chunk_size": chunk_size,
        "chunks": len(chunks),
        "hit_at_1": hits[1] / query_count,
        "hit_at_3": hits[3] / query_count,
        "hit_at_5": hits[5] / query_count,
        "hit_at_10": hits[10] / query_count,
        "mrr_at_10": reciprocal_rank_sum / query_count
    }

    experiment_results.append(result)

    print(
        f"Hit@1={result['hit_at_1']:.4f} | "
        f"Hit@3={result['hit_at_3']:.4f} | "
        f"Hit@5={result['hit_at_5']:.4f} | "
        f"Hit@10={result['hit_at_10']:.4f} | "
        f"MRR@10={result['mrr_at_10']:.4f}"
    )


output_path = "experiments/scifact_chunk_size_results.csv"

with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "chunk_size",
            "chunks",
            "hit_at_1",
            "hit_at_3",
            "hit_at_5",
            "hit_at_10",
            "mrr_at_10"
        ]
    )

    writer.writeheader()
    writer.writerows(experiment_results)


best_result = max(
    experiment_results,
    key=lambda result: result["mrr_at_10"]
)


print("\n" + "=" * 75)
print("BEST CONFIGURATION BY MRR@10")
print("=" * 75)

print(f"Chunk size: {best_result['chunk_size']}")
print(f"Chunks: {best_result['chunks']}")
print(f"Hit@1: {best_result['hit_at_1']:.4f}")
print(f"Hit@3: {best_result['hit_at_3']:.4f}")
print(f"Hit@5: {best_result['hit_at_5']:.4f}")
print(f"Hit@10: {best_result['hit_at_10']:.4f}")
print(f"MRR@10: {best_result['mrr_at_10']:.4f}")

print(f"\nResults saved to: {output_path}")