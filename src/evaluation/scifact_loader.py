from collections import defaultdict
from datasets import load_dataset


def load_scifact():
    corpus_dataset = load_dataset(
        "BeIR/scifact",
        "corpus",
        split="corpus"
    )

    query_dataset = load_dataset(
        "BeIR/scifact",
        "queries",
        split="queries"
    )

    qrels_dataset = load_dataset(
        "BeIR/scifact-qrels",
        split="test"
    )

    corpus = {}

    for document in corpus_dataset:
        document_id = str(document["_id"])

        corpus[document_id] = {
            "title": document["title"],
            "text": document["text"]
        }

    queries = {}

    for query in query_dataset:
        query_id = str(query["_id"])
        queries[query_id] = query["text"]

    qrels = defaultdict(set)

    for judgment in qrels_dataset:
        query_id = str(judgment["query-id"])
        corpus_id = str(judgment["corpus-id"])

        if judgment["score"] > 0:
            qrels[query_id].add(corpus_id)

    return corpus, queries, dict(qrels)