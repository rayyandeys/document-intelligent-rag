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


total_hits = 0
total_rr = 0.0
top_k = 5

print("\nRETRIEVAL EVALUATION")
print("=" * 70)


for item in evaluation_data:

    results = retriever.retrieve(
        item["question"],
        top_k=top_k
    )

    hit = hit_at_k(
        results,
        item["expected_pages"]
    )

    rr = reciprocal_rank(
        results,
        item["expected_pages"]
    )

    total_hits += hit
    total_rr += rr

    retrieved_pages = [
        result["page"]
        for result in results
    ]

    print(f"\nQuestion {item['id']}: {item['question']}")
    print(f"Expected pages: {item['expected_pages']}")
    print(f"Retrieved pages: {retrieved_pages}")
    print(f"Hit@{top_k}: {hit}")
    print(f"Reciprocal Rank: {rr:.4f}")


num_questions = len(evaluation_data)

hit_rate = total_hits / num_questions
mrr = total_rr / num_questions

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"Questions evaluated: {num_questions}")
print(f"Hit@{top_k}: {hit_rate:.4f}")
print(f"MRR: {mrr:.4f}")