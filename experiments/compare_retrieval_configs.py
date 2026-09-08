import csv
import json

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.smart_chunker import smart_chunk_pages
from src.retrieval.embedder import Embedder
from src.retrieval.retriever import Retriever
from src.evaluation.retrieval_eval import hit_at_k, reciprocal_rank


with open(
    "data/processed/evaluation_questions.json",
    "r",
    encoding="utf-8"
) as file:
    evaluation_data = json.load(file)


pages = load_pdf("data/raw/sample.pdf")
embedder = Embedder()

chunk_sizes = [400, 600, 800, 1000, 1200]
top_k_values = [1, 3, 5]

results_table = []


for chunk_size in chunk_sizes:

    chunks = smart_chunk_pages(
        pages,
        chunk_size=chunk_size,
        overlap_sentences=1
    )

    embeddings = embedder.embed_chunks(chunks)

    retriever = Retriever(
        chunks=chunks,
        embeddings=embeddings,
        embedder=embedder
    )

    for top_k in top_k_values:

        total_hits = 0
        total_rr = 0.0

        for item in evaluation_data:

            results = retriever.retrieve(
                item["question"],
                top_k=top_k
            )

            total_hits += hit_at_k(
                results,
                item["expected_pages"]
            )

            total_rr += reciprocal_rank(
                results,
                item["expected_pages"]
            )

        question_count = len(evaluation_data)

        hit_rate = total_hits / question_count
        mrr = total_rr / question_count

        result = {
            "chunk_size": chunk_size,
            "top_k": top_k,
            "chunks": len(chunks),
            "hit_rate": hit_rate,
            "mrr": mrr
        }

        results_table.append(result)

        print(
            f"Chunk size={chunk_size:<4} | "
            f"Top-k={top_k} | "
            f"Chunks={len(chunks):<3} | "
            f"Hit@k={hit_rate:.4f} | "
            f"MRR={mrr:.4f}"
        )


best_result = max(
    results_table,
    key=lambda result: (result["mrr"], result["hit_rate"])
)


print("\n" + "=" * 75)
print("BEST CONFIGURATION")
print("=" * 75)

print(f"Chunk size: {best_result['chunk_size']}")
print(f"Top-k: {best_result['top_k']}")
print(f"Chunks: {best_result['chunks']}")
print(f"Hit@k: {best_result['hit_rate']:.4f}")
print(f"MRR: {best_result['mrr']:.4f}")

output_path = "experiments/retrieval_results.csv"

with open(output_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "chunk_size",
            "top_k",
            "chunks",
            "hit_rate",
            "mrr"
        ]
    )

    writer.writeheader()
    writer.writerows(results_table)

print(f"\nResults saved to: {output_path}")