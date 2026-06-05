import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords", quiet=True)

import pandas as pd

from preprocessing.preprocessing import (
    load_qrels,
    load_smart,
    preprocess,
    preprocess_collection,
)
from query_expansion.query_expansion import QueryExpander
from document_retrieval.retrieval import retrieve_documents
from scoring.scoring_options import ScoringOptions, TFIDFScheme, TFWeight

DATA_DIR = "data"
MODEL_DIR = "models/pretrained_qegans"
TOP_K = 5

PREPROC_CONFIGS = [(stem, stop) for stem in (False, True) for stop in (False, True)]
TF_WEIGHTS = [
    ("raw tf", TFWeight.RAW_TF),
    ("binary tf", TFWeight.BINARY_TF),
    ("log tf", TFWeight.LOG_TF),
    ("augmented tf", TFWeight.AUGMENTED_TF),
]
NORM_SCHEMES = [("no", TFIDFScheme.RAW), ("yes", TFIDFScheme.NORMALIZED)]


def mean_ap(ap: dict[str, float], judged: list[str]) -> float:
    return sum(ap[q] for q in judged) / len(judged) if judged else 0.0


def run_benchmark(progress=None) -> pd.DataFrame:
    raw_docs = load_smart(f"{DATA_DIR}/cisi.all")
    raw_queries = load_smart(f"{DATA_DIR}/query.text", fields=("W",))
    qrels = load_qrels(f"{DATA_DIR}/qrels.text")
    judged = sorted(qrels)

    rows = []
    for stem, stop in PREPROC_CONFIGS:
        if progress:
            progress(f"Preprocessing: stem={stem}, remove_stopwords={stop} ...")
        processed_docs = preprocess_collection(
            raw_docs, stem=stem, remove_stopwords=stop
        )
        original = {
            q: preprocess(raw_queries[q], stem=stem, remove_stopwords=stop)
            for q in judged
        }

        expander = QueryExpander(
            use_stemming=stem, use_stopwords=stop, model_dir=MODEL_DIR
        )
        expanded = {}
        for q in judged:
            tokens = original[q]
            if tokens:
                terms = expander.expand(" ".join(tokens), top_k=TOP_K)
                expanded[q] = tokens + [t for t, _ in terms]
            else:
                expanded[q] = tokens

        for tf_name, tf in TF_WEIGHTS:
            for norm_name, scheme in NORM_SCHEMES:
                opts = ScoringOptions(
                    docs_tf_weight=tf,
                    docs_tf_idf_scheme=scheme,
                    query_tf_weight=tf,
                    query_tf_idf_scheme=scheme,
                )
                _, ap_o = retrieve_documents(
                    original, processed_docs, opts, ground_truths=qrels
                )
                _, ap_e = retrieve_documents(
                    expanded, processed_docs, opts, ground_truths=qrels
                )
                map_o = mean_ap(ap_o, judged)
                map_e = mean_ap(ap_e, judged)
                rows.append(
                    {
                        "stem": stem,
                        "stopwords": stop,
                        "tf_weight": tf_name,
                        "normalization": norm_name,
                        "map_original": round(map_o, 4),
                        "map_expanded": round(map_e, 4),
                        "delta": round(map_e - map_o, 4),
                    }
                )

    return pd.DataFrame(rows)
