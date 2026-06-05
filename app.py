import random
import time

import pandas as pd
import streamlit as st
import math
from collections import Counter
import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords', quiet=True)

from scoring.scoring_options import ScoringOptions, TFIDFScheme, TFWeight
from preprocessing.preprocessing import load_smart, preprocess, preprocess_collection
from inverted_index.inverted_index import InvertedIndex
from scoring.scoring import ScoringResults
from query_expansion.query_expansion import QueryExpander


# Page config
st.set_page_config(
    layout="wide",
    page_title="STBI Search Engine",
    initial_sidebar_state="expanded",
)


# Styling
st.markdown(
    """
    <style>
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        header[data-testid="stHeader"] { background: transparent; }

        [data-testid="stSidebarCollapsedControl"] {
            visibility: visible !important;
            display: block !important;
        }

        div[data-baseweb="select"] input {
            caret-color: transparent !important;
            pointer-events: none !important;
        }

        div[data-testid="InputInstructions"] { display: none !important; }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        body { background-color: #ebebeb; color: #334155; }

        /* App header */
        .app-title {
            font-size: 30px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0;
        }
        .app-subtitle {
            font-size: 15px;
            color: #64748b;
            font-weight: 500;
            margin-top: 2px;
            margin-bottom: 8px;
        }

        /* Metric card */
        .metric-card {
            background-color: #ffffff;
            padding: 20px 24px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            border-top: 4px solid #94a3b8;
            height: 100%;
            min-height: 135px;
            box-sizing: border-box;
        }
        .metric-card.accent {
            border-top: 4px solid #10b981;
        }
        .metric-title {
            font-size: 13px;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .metric-value {
            font-size: 36px;
            font-weight: 700;
            color: #0f172a;
        }
        .metric-delta {
            font-size: 13px;
            font-weight: 600;
            color: #059669;
        }

        /* Query badge */
        .query-badge {
            background-color: #f8fafc;
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 14px;
            margin-bottom: 20px;
            border: 1px solid #e2e8f0;
            color: #334155;
        }
        .query-badge .label {
            font-size: 11px;
            text-transform: uppercase;
            color: #94a3b8;
            font-weight: 700;
            display: block;
            margin-bottom: 6px;
            letter-spacing: 0.04em;
        }
        .highlight {
            color: #065f46;
            font-weight: 700;
            background-color: #d1fae5;
            padding: 2px 6px;
            border-radius: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = "data"
MODEL_DIR = "models/pretrained_qegans"


@st.cache_resource(show_spinner=False)
def get_query_expander(use_stemming: bool, remove_stopwords: bool):
    """Load (once) the pretrained GAN QueryExpander for this preprocessing config.

    NOTE: the backend's parameter is named `use_stopwords`; we map the UI's
    `remove_stopwords` flag straight onto it so the prefix
    `stem_{use_stemming}_stop_{remove_stopwords}` selects the matching model.
    Confirm this semantic mapping is correct during Phase 3 wiring.
    """
    return QueryExpander(
        use_stemming=use_stemming,
        use_stopwords=remove_stopwords,
        model_dir=MODEL_DIR,
    )


@st.cache_resource(show_spinner=False)
def get_retrieval_system(use_stemming: bool, remove_stopwords: bool,
                         tf_weight: TFWeight, tfidf_scheme: TFIDFScheme):
    """Load the CISI corpus and build the inverted index + scoring tables once.

    Cached on the primitive, hashable settings that actually change the corpus or
    the weights (the raw `settings` dict carries a ScoringOptions object that
    Streamlit can't hash, so we pass the individual fields instead). Returns a
    dict the retrieval step can reuse without rebuilding the index every search.
    """
    processed_docs = preprocess_collection(
        load_smart(f"{DATA_DIR}/cisi.all"),
        stem=use_stemming,
        remove_stopwords=remove_stopwords,
    )
    inverted_index = InvertedIndex(processed_docs)
    scoring_options = ScoringOptions(
        docs_tf_weight=tf_weight,
        docs_tf_idf_scheme=tfidf_scheme,
        query_tf_weight=tf_weight,
        query_tf_idf_scheme=tfidf_scheme,
    )
    scoring_results = ScoringResults(
        inverted_index, len(processed_docs), scoring_options
    )
    return {
        "processed_docs": processed_docs,
        "inverted_index": inverted_index,
        "scoring_results": scoring_results,
        "scoring_options": scoring_options,
    }

def _rank_query(tokens, system):
    """Rank documents for query tokens using the cached scoring tables.

    Mirrors document_retrieval.retrieval.retrieve_documents()'s per-query scoring,
    but reuses the cached InvertedIndex / ScoringResults instead of rebuilding the
    index on every search. Returns [(doc_id, score), ...] sorted best-first.
    """
    scoring = system["scoring_results"]
    options = system["scoring_options"]
    idf = scoring.idf
    docs_weight = scoring.tf_idf
    doc_lengths = scoring.doc_lengths

    term_tf = Counter(tokens)
    q_max_tf = max(term_tf.values()) if term_tf else 1

    scores = {}
    query_length = 0.0
    for term, q_tf in term_tf.items():
        if term not in idf:
            continue
        if options.query_tf_weight == TFWeight.RAW_TF:
            q_weight = q_tf * idf[term]
        elif options.query_tf_weight == TFWeight.LOG_TF:
            q_weight = (1 + math.log2(q_tf)) * idf[term]
        elif options.query_tf_weight == TFWeight.BINARY_TF:
            q_weight = 1 * idf[term]
        else: 
            q_weight = (0.5 + 0.5 * q_tf / q_max_tf) * idf[term]

        query_length += q_weight ** 2
        for doc_id, d_weight in docs_weight[term].items():
            scores[doc_id] = scores.get(doc_id, 0.0) + q_weight * d_weight

    if options.docs_tf_idf_scheme == TFIDFScheme.NORMALIZED:
        for doc_id in scores:
            d_norm = doc_lengths[doc_id] if doc_lengths[doc_id] > 0 else 1
            scores[doc_id] /= d_norm
    if options.query_tf_idf_scheme == TFIDFScheme.NORMALIZED:
        q_norm = math.sqrt(query_length) if query_length > 0 else 1
        for doc_id in scores:
            scores[doc_id] /= q_norm

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def run_expand_query(query, settings):
    """Real GAN query expansion. Returns [(term, weight), ...]."""
    tokens = preprocess(
        query,
        stem=settings["use_stemming"],
        remove_stopwords=settings["remove_stopwords"],
    )
    if not tokens:
        return []
    expander = get_query_expander(
        settings["use_stemming"], settings["remove_stopwords"]
    )
    return expander.expand(
        " ".join(tokens),
        top_k=settings["expansion_limit"],
        return_all=settings["add_all_words"],
    )


def run_search(query, settings):
    """Run retrieval for both the original and the GAN-expanded query."""
    system = get_retrieval_system(
        settings["use_stemming"],
        settings["remove_stopwords"],
        settings["tf_weight"],
        settings["tfidf_scheme"],
    )
    tokens = preprocess(
        query,
        stem=settings["use_stemming"],
        remove_stopwords=settings["remove_stopwords"],
    )
    expanded_terms = run_expand_query(query, settings)
    expanded_tokens = tokens + [term for term, _ in expanded_terms]

    return {
        "original_query": query,
        "expanded_terms": expanded_terms,
        "map_original": 0.0,
        "map_expanded": 0.0,
        "ranking_original": format_ranking_to_df(_rank_query(tokens, system)),
        "ranking_expanded": format_ranking_to_df(_rank_query(expanded_tokens, system)),
    }


def get_index_stats(doc_id, system):
    """Pull inverted-index entries for one document from the cached index.

    Returns (stats, dataframe): stats holds corpus totals, and the DataFrame lists
    every term occurring in the requested document with its TF / DF / IDF / TF-IDF.
    """
    inverted_index = system["inverted_index"].inverted_index
    scoring = system["scoring_results"]
    doc_id = str(doc_id)

    rows = []
    for term, postings in inverted_index.items():
        if doc_id in postings:
            rows.append(
                {
                    "Term": term,
                    "Term Frequency": postings[doc_id],
                    "Document Frequency": len(postings),
                    "IDF": round(scoring.idf[term], 4),
                    "TF-IDF Weight": round(scoring.tf_idf[term][doc_id], 4),
                }
            )
    rows.sort(key=lambda r: r["Term Frequency"], reverse=True)
    df = pd.DataFrame(
        rows,
        columns=["Term", "Term Frequency", "Document Frequency", "IDF", "TF-IDF Weight"],
    )
    stats = {
        "total_terms": len(inverted_index),
        "total_docs": len(system["processed_docs"]),
        "doc_term_count": len(rows),
    }
    return stats, df


def mock_process_batch(file_bytes):
    """Pretend to process a batch query file and return mock artifacts."""
    text = file_bytes.decode("utf-8", errors="ignore")
    queries = [ln.strip() for ln in text.splitlines() if ln.strip()]

    map_lines = ["query_id\tMAP_original\tMAP_expanded"]
    retrieval_lines = ["query_id\trank\tdoc_id\tsimilarity"]
    random.seed(42)
    for i, _ in enumerate(queries, start=1):
        map_lines.append(
            f"{i}\t{round(random.uniform(0.2, 0.45), 4)}\t{round(random.uniform(0.45, 0.7), 4)}"
        )
        score = 0.95
        for rank in range(1, 11):
            retrieval_lines.append(
                f"{i}\t{rank}\t{random.randint(1, 1488)}\t{round(score, 4)}"
            )
            score -= random.uniform(0.02, 0.08)

    mean_original = round(random.uniform(0.25, 0.40), 4)
    mean_expanded = round(random.uniform(0.45, 0.65), 4)
    map_lines.append(f"ALL\t{mean_original}\t{mean_expanded}")

    return {
        "num_queries": len(queries),
        "map_file": "\n".join(map_lines),
        "retrieval_file": "\n".join(retrieval_lines),
        "mean_original": mean_original,
        "mean_expanded": mean_expanded,
    }


# Backend adapters
def format_ranking_to_df(raw_ranking):
    """Convert a raw ranking from the backend into the DataFrame the UI expects.

    Input: list of (doc_id, score) tuples, already sorted best-first, as returned
    by document_retrieval.retrieval.retrieve_documents()[qid].
    Output: DataFrame with columns ["Rank", "Document ID", "Similarity"].
    """
    rows = [
        {"Rank": rank, "Document ID": str(doc_id), "Similarity": round(float(score), 4)}
        for rank, (doc_id, score) in enumerate(raw_ranking, start=1)
    ]
    return pd.DataFrame(rows, columns=["Rank", "Document ID", "Similarity"])


# HTML render helpers
def render_metric_card(title, value, accent=False, delta=""):
    accent_cls = " accent" if accent else ""
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="metric-card{accent_cls}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_query_badge(label, query, expanded_terms=None):
    body = query
    if expanded_terms:
        max_terms = 15
        shown_terms = expanded_terms[:max_terms]
        highlights = " ".join(
            f'<span class="highlight">{term}</span>' for term, _ in shown_terms
        )
        body = f"{query} {highlights}"
        remaining = len(expanded_terms) - len(shown_terms)
        if remaining > 0:
            body += f' <span class="label">...and {remaining} more terms</span>'
    st.markdown(
        f"""
        <div class="query-badge">
            <span class="label">{label}</span>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


# Sidebar
with st.sidebar:
    st.markdown("### Settings & Filters")

    st.markdown("#### Preprocessing")
    use_stemming = st.checkbox("Stemming", value=True)
    remove_stopwords = st.checkbox("Remove Stop-words", value=True)

    st.markdown("#### Weighting")
    weighting_method = st.selectbox(
        "Weighting Method",
        options=[
            "TF (logarithmic/binary/augmented/raw)",
            "IDF Only",
            "TF-IDF",
            "TF-IDF + Cosine Normalization",
        ],
        index=3,
    )
    tf_variant = None
    if weighting_method.startswith("TF ("):
        tf_variant = st.selectbox(
            "TF Variant",
            options=["Raw", "Logarithmic", "Binary", "Augmented"],
            index=0,
        )

    st.markdown("#### Query Expansion")
    add_all_words = st.checkbox("Add All Words", value=False)
    expansion_limit = st.number_input(
        "Expansion Word Limit",
        min_value=0,
        max_value=100,
        value=5,
        step=1,
        disabled=add_all_words,
        help="Number of expansion terms to add. Ignored when 'Add All Words' is checked.",
    )

_TF_VARIANT_TO_ENUM = {
    "Raw": TFWeight.RAW_TF,
    "Logarithmic": TFWeight.LOG_TF,
    "Binary": TFWeight.BINARY_TF,
    "Augmented": TFWeight.AUGMENTED_TF,
}

if weighting_method.startswith("TF ("):
    tf_weight = _TF_VARIANT_TO_ENUM.get(tf_variant, TFWeight.RAW_TF)
    tfidf_scheme = TFIDFScheme.RAW
elif weighting_method == "IDF Only":
    tf_weight = TFWeight.BINARY_TF
    tfidf_scheme = TFIDFScheme.RAW
elif weighting_method == "TF-IDF":
    tf_weight = TFWeight.RAW_TF
    tfidf_scheme = TFIDFScheme.RAW
else:  # TF-IDF + Cosine Normalization
    tf_weight = TFWeight.RAW_TF
    tfidf_scheme = TFIDFScheme.NORMALIZED

scoring_options = ScoringOptions(
    docs_tf_weight=tf_weight,
    docs_tf_idf_scheme=tfidf_scheme,
    query_tf_weight=tf_weight,
    query_tf_idf_scheme=tfidf_scheme,
)

settings = {
    "use_stemming": use_stemming,
    "remove_stopwords": remove_stopwords,
    "weighting_method": weighting_method,
    "tf_variant": tf_variant,
    "add_all_words": add_all_words,
    "expansion_limit": int(expansion_limit),
    "tf_weight": tf_weight,
    "tfidf_scheme": tfidf_scheme,
    "scoring_options": scoring_options,
}


# Main area header
st.markdown('<div class="app-title">Retrieval Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Advanced document search powered by GAN query expansion.</div>',
    unsafe_allow_html=True,
)

interactive_tab, batch_tab, index_tab = st.tabs(
    ["Interactive Search", "Batch Processing", "Inverted Index Inspector"]
)


# Tab 1: Interactive Search
with interactive_tab:
    with st.form("search_form"):
        search_col, button_col = st.columns([5, 1])
        with search_col:
            query = st.text_input(
                "Search query",
                placeholder="e.g. automatic information retrieval from titles",
                label_visibility="collapsed",
            )
        with button_col:
            search_clicked = st.form_submit_button(
                "Search", type="primary", use_container_width=True
            )

    if search_clicked:
        if not query.strip():
            st.warning("Please enter a search query first.")
        else:
            with st.spinner("Running GAN model and expanding query..."):
                results = run_search(query, settings)

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                render_metric_card(
                    "MAP - Original Query", f"{results['map_original']:.4f}"
                )
            with metric_col2:
                delta = results["map_expanded"] - results["map_original"]
                render_metric_card(
                    "MAP - Expanded Query",
                    f"{results['map_expanded']:.4f}",
                    accent=True,
                    delta=f"+{delta:.4f} vs original",
                )
            st.write("")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.subheader("Original Query Results")
                render_query_badge("Original Query", results["original_query"])
                st.dataframe(
                    results["ranking_original"],
                    use_container_width=True,
                    hide_index=True,
                )
            with res_col2:
                st.subheader("Expanded Query Results")
                render_query_badge(
                    "Expanded Query (GAN)",
                    results["original_query"],
                    expanded_terms=results["expanded_terms"],
                )
                st.dataframe(
                    results["ranking_expanded"],
                    use_container_width=True,
                    hide_index=True,
                )

            if results["expanded_terms"]:
                with st.expander("Expansion terms and weights"):
                    st.dataframe(
                        pd.DataFrame(
                            results["expanded_terms"], columns=["Added Term", "Weight"]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )


# Tab 2: Batch Processing
with batch_tab:
    st.subheader("Batch Processing")
    st.caption(
        "Upload a .txt file containing multiple queries (one query per line). "
        "Results and MAP rankings will be available for download."
    )

    uploaded_file = st.file_uploader("Upload queries file", type=["txt"])
    process_clicked = st.button("Process Batch", type="primary", key="batch_btn")

    if process_clicked:
        if uploaded_file is None:
            st.warning("Please upload a .txt query file first.")
        else:
            file_bytes = uploaded_file.getvalue()
            if not file_bytes.decode("utf-8", errors="ignore").strip():
                st.error(
                    "The uploaded file is empty. Please provide a valid query file."
                )
                st.stop()

            with st.spinner("Processing batch queries and calculating MAP..."):
                time.sleep(1.5)
                batch_result = mock_process_batch(file_bytes)
            st.success(f"Processed {batch_result['num_queries']} queries successfully.")

            mean_col1, mean_col2 = st.columns(2)
            with mean_col1:
                render_metric_card(
                    "Mean MAP - Original", f"{batch_result['mean_original']:.4f}"
                )
            with mean_col2:
                delta = batch_result["mean_expanded"] - batch_result["mean_original"]
                render_metric_card(
                    "Mean MAP - Expanded",
                    f"{batch_result['mean_expanded']:.4f}",
                    accent=True,
                    delta=f"+{delta:.4f} vs original",
                )

            st.write("")
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "Download MAP Results",
                    data=batch_result["map_file"],
                    file_name="map_results.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with dl_col2:
                st.download_button(
                    "Download Retrieval Results",
                    data=batch_result["retrieval_file"],
                    file_name="retrieval_results.txt",
                    mime="text/plain",
                    use_container_width=True,
                )


# Tab 3: Inverted Index Inspector
with index_tab:
    st.subheader("Inverted Index Inspector")
    st.caption("Inspect the inverted index entries for a specific document.")

    with st.form("index_form"):
        idx_col1, idx_col2 = st.columns([5, 1])
        with idx_col1:
            inspect_doc_id = st.text_input(
                "Document ID",
                value="1",
                label_visibility="collapsed",
                placeholder="Enter a document ID (e.g. 1)",
            )
        with idx_col2:
            inspect_clicked = st.form_submit_button(
                "Search", type="primary", use_container_width=True
            )

    if inspect_clicked:
        if not inspect_doc_id.strip():
            st.warning("Please enter a document ID first.")
        else:
            system = get_retrieval_system(
                settings["use_stemming"],
                settings["remove_stopwords"],
                settings["tf_weight"],
                settings["tfidf_scheme"],
            )
            stats, index_df = get_index_stats(inspect_doc_id, system)

            stat_col1, stat_col2, stat_col3 = st.columns(3)
            stat_col1.metric("Total Terms (corpus)", f"{stats['total_terms']:,}")
            stat_col2.metric("Total Documents", f"{stats['total_docs']:,}")
            stat_col3.metric("Terms in This Document", f"{stats['doc_term_count']:,}")

            render_query_badge("Document", inspect_doc_id.strip())
            if index_df.empty:
                st.info(
                    f"No inverted-index entries found for document "
                    f"'{inspect_doc_id.strip()}'. Check the document ID."
                )
            else:
                st.dataframe(
                    index_df,
                    use_container_width=True,
                    hide_index=True,
                )