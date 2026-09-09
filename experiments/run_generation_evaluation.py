import csv
import json
import time
from pathlib import Path

from google.genai.errors import ClientError, ServerError

from src.evaluation.generation_eval import evaluate_generation_result
from src.generation.generator import Generator
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.smart_chunker import smart_chunk_pages
from src.retrieval.embedder import Embedder
from src.retrieval.faiss_retriever import FAISSRetriever


PDF_PATH = "data/raw/sample.pdf"
QUESTIONS_PATH = "data/processed/generation_evaluation_questions.json"
OUTPUT_PATH = "experiments/generation_evaluation_results.csv"

CHUNK_SIZE = 400
OVERLAP_SENTENCES = 1
TOP_K = 5

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10


def save_results(results):
    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "question",
                "expect_refusal",
                "is_refusal",
                "has_citation",
                "correct_behavior",
                "top_retrieval_score",
                "answer"
            ]
        )

        writer.writeheader()
        writer.writerows(results)


print("=" * 75)
print("CONTROLLED GENERATION / GROUNDING EVALUATION")
print("=" * 75)

# ---------------------------------------------------------------------
# Load document
# ---------------------------------------------------------------------

pages = load_pdf(PDF_PATH)

for page in pages:
    page["source"] = "sample.pdf"

chunks = smart_chunk_pages(
    pages,
    chunk_size=CHUNK_SIZE,
    overlap_sentences=OVERLAP_SENTENCES
)

print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Top-k: {TOP_K}")

# ---------------------------------------------------------------------
# Build retrieval pipeline once
# ---------------------------------------------------------------------

print("\nLoading embedding model...")

embedder = Embedder()

print("Embedding document chunks...")

embeddings = embedder.embed_chunks(chunks)

retriever = FAISSRetriever(
    chunks=chunks,
    embeddings=embeddings,
    embedder=embedder
)

generator = Generator()

# ---------------------------------------------------------------------
# Load questions
# ---------------------------------------------------------------------

with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8"
) as file:
    questions = json.load(file)

# ---------------------------------------------------------------------
# Resume existing results if present
# ---------------------------------------------------------------------

results = []
completed_ids = set()

if Path(OUTPUT_PATH).exists():

    with open(
        OUTPUT_PATH,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            results.append(row)
            completed_ids.add(int(row["id"]))

    print(
        f"\nResuming evaluation: "
        f"{len(completed_ids)} completed question(s) found."
    )

# ---------------------------------------------------------------------
# Run remaining questions
# ---------------------------------------------------------------------

for item in questions:

    question_id = item["id"]

    if question_id in completed_ids:
        print(f"\nSkipping Question {question_id}: already completed.")
        continue

    question = item["question"]
    expect_refusal = item["expect_refusal"]

    print("\n" + "-" * 75)
    print(f"Question {question_id}: {question}")

    retrieved_chunks = retriever.retrieve(
        question,
        top_k=TOP_K
    )

    answer = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            answer = generator.generate(
                query=question,
                retrieved_chunks=retrieved_chunks
            )
            break

        except (ServerError, ClientError) as error:

            print(
                f"Gemini server error on attempt "
                f"{attempt}/{MAX_RETRIES}: {error}"
            )

            if attempt < MAX_RETRIES:
                print(
                    f"Waiting {RETRY_DELAY_SECONDS} seconds "
                    f"before retry..."
                )
                time.sleep(RETRY_DELAY_SECONDS)

    if answer is None:
        print(
            f"Question {question_id} could not be completed "
            f"after {MAX_RETRIES} attempts."
        )
        print(
            "Stopping safely. Completed results remain saved."
        )
        break

    evaluation = evaluate_generation_result(
        answer=answer,
        expect_refusal=expect_refusal
    )

    top_score = (
        float(retrieved_chunks[0]["score"])
        if retrieved_chunks
        else 0.0
    )

    result = {
        "id": question_id,
        "question": question,
        "expect_refusal": expect_refusal,
        "is_refusal": evaluation["is_refusal"],
        "has_citation": evaluation["has_citation"],
        "correct_behavior": evaluation["correct_behavior"],
        "top_retrieval_score": top_score,
        "answer": answer
    }

    results.append(result)
    completed_ids.add(question_id)

    # Save immediately so a later API failure cannot lose this result.
    save_results(results)

    print(f"Expected refusal: {expect_refusal}")
    print(f"Actual refusal: {evaluation['is_refusal']}")
    print(f"Citation present: {evaluation['has_citation']}")
    print(f"Correct behavior: {evaluation['correct_behavior']}")
    print(f"Top retrieval score: {top_score:.4f}")
    print(f"Answer: {answer}")

# ---------------------------------------------------------------------
# Calculate summary from completed results
# ---------------------------------------------------------------------

supported_rows = [
    row for row in results
    if str(row["expect_refusal"]).lower() == "false"
]

unsupported_rows = [
    row for row in results
    if str(row["expect_refusal"]).lower() == "true"
]

supported_success = sum(
    str(row["correct_behavior"]).lower() == "true"
    for row in supported_rows
)

unsupported_success = sum(
    str(row["correct_behavior"]).lower() == "true"
    for row in unsupported_rows
)

overall_success = sum(
    str(row["correct_behavior"]).lower() == "true"
    for row in results
)

print("\n" + "=" * 75)
print("GENERATION EVALUATION SUMMARY")
print("=" * 75)

if supported_rows:
    print(
        f"Supported citation behavior: "
        f"{supported_success}/{len(supported_rows)} "
        f"({supported_success / len(supported_rows):.2%})"
    )

if unsupported_rows:
    print(
        f"Unsupported refusal behavior: "
        f"{unsupported_success}/{len(unsupported_rows)} "
        f"({unsupported_success / len(unsupported_rows):.2%})"
    )

if results:
    print(
        f"Overall controlled behavior: "
        f"{overall_success}/{len(results)} "
        f"({overall_success / len(results):.2%})"
    )

print(
    f"Completed questions: "
    f"{len(results)}/{len(questions)}"
)

print(f"\nResults saved to: {OUTPUT_PATH}")