from datasets import load_dataset


corpus = load_dataset(
    "BeIR/scifact",
    "corpus",
    split="corpus"
)

queries = load_dataset(
    "BeIR/scifact",
    "queries",
    split="queries"
)

qrels = load_dataset(
    "BeIR/scifact-qrels",
    split="test"
)


print("=" * 60)
print("SCIFACT DATASET")
print("=" * 60)

print(f"Documents: {len(corpus)}")
print(f"Queries: {len(queries)}")
print(f"Qrels: {len(qrels)}")

print("\nSample document:")
print(corpus[0])

print("\nSample query:")
print(queries[0])

print("\nSample relevance judgment:")
print(qrels[0])