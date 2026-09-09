INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The provided documents do not contain enough information "
    "to answer this question."
)


def evaluate_generation_result(
    answer: str,
    expect_refusal: bool
) -> dict:
    normalized_answer = answer.strip()

    is_refusal = (
        normalized_answer == INSUFFICIENT_EVIDENCE_MESSAGE
    )

    has_citation = (
        "[Source:" in answer and
        "Page:" in answer
    )

    if expect_refusal:
        correct_behavior = is_refusal
    else:
        correct_behavior = (
            not is_refusal and
            has_citation
        )

    return {
        "is_refusal": is_refusal,
        "has_citation": has_citation,
        "correct_behavior": correct_behavior
    }