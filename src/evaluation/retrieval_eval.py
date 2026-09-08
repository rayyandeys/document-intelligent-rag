from typing import List, Dict


def hit_at_k(results: List[Dict], expected_pages: List[int]) -> int:
    retrieved_pages = [result["page"] for result in results]

    for page in expected_pages:
        if page in retrieved_pages:
            return 1

    return 0


def reciprocal_rank(results: List[Dict], expected_pages: List[int]) -> float:
    for rank, result in enumerate(results, start=1):
        if result["page"] in expected_pages:
            return 1.0 / rank

    return 0.0