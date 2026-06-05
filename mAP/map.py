from collections import defaultdict

def calculate_average_precision(ranked_docs: list[tuple[str, float]], relevant_docs: set[str]) -> float:
    """Return Average Precision for a single ranked query."""
    if not relevant_docs:
        return 0.0

    num_relevant = 0
    precision_sum = 0.0
    for rank, (doc_id, _) in enumerate(ranked_docs, start=1):
        if doc_id in relevant_docs:
            num_relevant += 1
            precision_sum += num_relevant / rank

    return precision_sum / len(relevant_docs)


def calculate_mean_average_precision(
    ranked_docs: dict[str, list[tuple[str, float]]],
    ground_truths: dict[str, set[str]],
) -> float:
    """Return mean Average Precision across all queries in ranked_docs."""
    average_precisions = []
    for qid, ranked in ranked_docs.items():
        relevant_docs = ground_truths.get(qid, set())
        if not relevant_docs:
            continue
        average_precisions.append(calculate_average_precision(ranked, relevant_docs))

    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0