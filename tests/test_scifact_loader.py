from src.evaluation.scifact_loader import load_scifact


corpus, queries, qrels = load_scifact()

print("=" * 60)
print("SCIFACT LOADER TEST")
print("=" * 60)

print(f"Documents loaded: {len(corpus)}")
print(f"Queries loaded: {len(queries)}")
print(f"Evaluated queries: {len(qrels)}")

sample_query_id = next(iter(qrels))

print("\nSample evaluated query:")
print("Query ID:", sample_query_id)
print("Question:", queries[sample_query_id])
print("Relevant document IDs:", qrels[sample_query_id])

for document_id in qrels[sample_query_id]:
    print("\nRelevant document:")
    print("ID:", document_id)
    print("Title:", corpus[document_id]["title"])