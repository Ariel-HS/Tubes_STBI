from collections import defaultdict
import math
from inverted_index.inverted_index import InvertedIndex
from scoring.scoring import ScoringResults
from scoring.scoring_options import ScoringOptions, TFIDFScheme, TFWeight

# Retrieves and ranks documents based on cosine similarity between query and document vectors
# Input: 
# - Query tokens (Dict of List of String), 
# - Processed documents (Dict of List of String), 
# - Scoring options (TF Weighting and TF-IDF Scheme)
# - ground_truths (optional): Dict of qid -> relevant doc IDs
# Return: 
# - Ranked Documents for each query
# - If ground_truths is provided, also returns per-query AP as a second value
#   and preserves backward compatibility when ground_truths is None.
def retrieve_documents(
    query_tokens: dict[str, list[str]],
    processed_docs: dict[str, list[str]],
    scoring_options=ScoringOptions(
        docs_tf_weight=TFWeight.RAW_TF, 
        docs_tf_idf_scheme=TFIDFScheme.NORMALIZED, 
        query_tf_weight=TFWeight.RAW_TF, 
        query_tf_idf_scheme=TFIDFScheme.NORMALIZED),
    ground_truths: dict[str, set[str]] | None = None,
) -> dict[str, list[tuple[str, float]]] | tuple[dict[str, list[tuple[str, float]]], dict[str, float]]:
    # Calculate term frequencies within the query
    queries_tf = defaultdict(lambda: defaultdict(int))
    queries_max_tf = defaultdict(int)

    for qid, tokens in query_tokens.items():
        for token in tokens:
            queries_tf[qid][token] += 1
            queries_tf["__BATCH__"][token] += 1

            if (queries_tf[qid][token] > queries_max_tf[qid]):
                queries_max_tf[qid] = queries_tf[qid][token]
            
            if (queries_tf["__BATCH__"][token] > queries_max_tf["__BATCH__"]):
                queries_max_tf["__BATCH__"] = queries_tf["__BATCH__"][token]

    inverted_index_object = InvertedIndex(processed_docs)
    inverted_index, _ = inverted_index_object.inverted_index, inverted_index_object.max_tf
    
    scoring_results = ScoringResults(inverted_index_object, len(processed_docs), scoring_options)
    idf, docs_weight, doc_lengths = scoring_results.idf, scoring_results.tf_idf, scoring_results.doc_lengths

    ranked_docs = {}
    query_average_precision: dict[str, float] = {}

    for qid, terms in queries_tf.items():
        scores = defaultdict(float)
        query_length = 0
        q_max_tf = queries_max_tf[qid]

        for term, q_tf in terms.items():
            if term not in inverted_index:
                continue  
            
            # Calculate weight based on scheme
            if scoring_options.query_tf_weight == TFWeight.RAW_TF:
                q_weight = q_tf * idf[term]
            elif scoring_options.query_tf_weight == TFWeight.LOG_TF:
                q_weight = (1 + math.log2(q_tf)) * idf[term]
            elif scoring_options.query_tf_weight == TFWeight.BINARY_TF:
                q_weight = 1 * idf[term]
            elif scoring_options.query_tf_weight == TFWeight.AUGMENTED_TF:
                q_weight = (0.5 + 0.5 * q_tf / q_max_tf) * idf[term]
            
            query_length += q_weight ** 2

            for doc_id, d_weight in docs_weight[term].items():
                scores[doc_id] += q_weight * d_weight
        
        # Normalize documents
        if scoring_options.docs_tf_idf_scheme == TFIDFScheme.NORMALIZED:
            for doc_id in scores:
                d_norm = doc_lengths[doc_id] if doc_lengths[doc_id] > 0 else 1
                scores[doc_id] /= d_norm

        # Normalize Query
        if scoring_options.query_tf_idf_scheme == TFIDFScheme.NORMALIZED:
            for doc_id in scores:
                q_norm = math.sqrt(query_length) if query_length > 0 else 1
                scores[doc_id] /= q_norm

        # Sort scores in descending order
        ranked_docs[qid] = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if ground_truths is not None and qid != "__BATCH__":
            query_average_precision[qid] = calculate_average_precision(
                ranked_docs[qid], ground_truths.get(qid, set())
            )
    
    if ground_truths is not None:
        return ranked_docs, query_average_precision

    return ranked_docs

def calculate_average_precision(ranked_docs: list[tuple[str, float]], relevant_docs: set[str]) -> float:
    """
Return Average Precision for a single ranked query.
If there are no relevant documents, return 0.0 to avoid division by zero.
Input:
- ranked_docs: List of tuples (doc_id, score) sorted by score in descending order
- relevant_docs: Set of relevant document IDs for the query
Output:
- Average Precision (float) for the query
    """
    if not relevant_docs:
        return 0.0

    num_relevant = 0
    precision_sum = 0.0
    for rank, (doc_id, _) in enumerate(ranked_docs, start=1):
        if doc_id in relevant_docs:
            num_relevant += 1
            precision_sum += num_relevant / rank

    return precision_sum / len(relevant_docs)