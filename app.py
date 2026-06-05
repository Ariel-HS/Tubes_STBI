import os
import tempfile

import pandas as pd
import streamlit as st
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
from document_retrieval.retrieval import retrieve_documents
from document_retrieval.retrieval import retrieve_documents


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
_WEIGHT_TO_ENUM = {
    "raw tf": TFWeight.RAW_TF,
    "binary tf": TFWeight.BINARY_TF,
    "log tf": TFWeight.LOG_TF,
    "augmented tf": TFWeight.AUGMENTED_TF,
}
_NORM_TO_ENUM = {
    "no": TFIDFScheme.RAW,
    "yes": TFIDFScheme.NORMALIZED,
}


@st.cache_resource(show_spinner=False)
def get_query_expander(use_stemming: bool, remove_stopwords: bool):
    return QueryExpander(
        use_stemming=use_stemming,
        use_stopwords=remove_stopwords,
        model_dir=MODEL_DIR,
    )


@st.cache_resource(show_spinner=False)
def get_retrieval_system(use_stemming: bool, remove_stopwords: bool,
                         doc_tf_weight: TFWeight, doc_tfidf_scheme: TFIDFScheme):
    raw_texts = load_smart(f"{DATA_DIR}/cisi.all")
    processed_docs = preprocess_collection(
        raw_texts,
        stem=use_stemming,
        remove_stopwords=remove_stopwords,
    )
    inverted_index = InvertedIndex(processed_docs)
    scoring_options = ScoringOptions(
        docs_tf_weight=doc_tf_weight,
        docs_tf_idf_scheme=doc_tfidf_scheme,
        query_tf_weight=doc_tf_weight,
        query_tf_idf_scheme=doc_tfidf_scheme,
    )
    scoring_results = ScoringResults(
        inverted_index, len(processed_docs), scoring_options
    )
    return {
        "raw_texts": raw_texts,
        "processed_docs": processed_docs,
        "inverted_index": inverted_index,
        "scoring_results": scoring_results,
        "scoring_options": scoring_options,
    }

def _build_scoring_options(settings):
    return ScoringOptions(
        docs_tf_weight=_WEIGHT_TO_ENUM[settings["doc_tf_variant"]],
        docs_tf_idf_scheme=_NORM_TO_ENUM[settings["doc_norm"]],
        query_tf_weight=_WEIGHT_TO_ENUM[settings["query_tf_variant"]],
        query_tf_idf_scheme=_NORM_TO_ENUM[settings["query_norm"]],
    )


def run_expand_query(query, settings):
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


def run_search(query, settings, expanded_terms=None):
    system = get_retrieval_system(
        settings["use_stemming"],
        settings["remove_stopwords"],
        _WEIGHT_TO_ENUM[settings["doc_tf_variant"]],
        _NORM_TO_ENUM[settings["doc_norm"]],
    )
    tokens = preprocess(
        query,
        stem=settings["use_stemming"],
        remove_stopwords=settings["remove_stopwords"],
    )
    if expanded_terms is None:
        expanded_terms = run_expand_query(query, settings)
    expanded_tokens = tokens + [term for term, _ in expanded_terms]

    rank_original = retrieve_documents({"q": tokens}, system["processed_docs"])["q"]
    rank_expanded = retrieve_documents({"q": expanded_tokens}, system["processed_docs"])["q"]

    rank_original = retrieve_documents({"q": tokens}, system["processed_docs"])["q"]
    rank_expanded = retrieve_documents({"q": expanded_tokens}, system["processed_docs"])["q"]

    ranked = retrieve_documents(
        {"__original__": tokens, "__expanded__": expanded_tokens},
        system["processed_docs"],
        _build_scoring_options(settings),
    )
    return {
        "original_query": query,
        "expanded_terms": expanded_terms,
        "map_original": 0.0,
        "map_expanded": 0.0,
        "ranking_original": format_ranking_to_df(rank_original),
        "ranking_expanded": format_ranking_to_df(rank_expanded),
    }


def get_index_stats(doc_id, system):
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


def _parse_batch_queries(text):
    if ".I" in text:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".text", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(text)
                tmp_path = tmp.name
            return load_smart(tmp_path, fields=("W",))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return {str(i): q for i, q in enumerate(lines, start=1)}


def run_batch_processing(uploaded_file, settings):
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    queries = _parse_batch_queries(text)

    system = get_retrieval_system(
        settings["use_stemming"],
        settings["remove_stopwords"],
        _WEIGHT_TO_ENUM[settings["doc_tf_variant"]],
        _NORM_TO_ENUM[settings["doc_norm"]],
    )
    scoring_options = _build_scoring_options(settings)
    original_tokens = {}
    expanded_tokens = {}
    expansion_counts = {}
    for qid, query_str in queries.items():
        tokens = preprocess(
            query_str,
            stem=settings["use_stemming"],
            remove_stopwords=settings["remove_stopwords"],
        )
        expanded_terms = run_expand_query(query_str, settings)
        expanded_tokens = tokens + [term for term, _ in expanded_terms]

        ranking_original = retrieve_documents({qid: tokens}, system["processed_docs"])[qid]
        ranking_expanded = retrieve_documents({qid: expanded_tokens}, system["processed_docs"])[qid]

        summary_rows.append(
            {
                "Query ID": qid,
                "Query": (query_str[:60] + "...") if len(query_str) > 60 else query_str,
                "Expansion Terms": expansion_counts[qid],
                "Docs Retrieved (Original)": len(ranked_original.get(qid, [])),
                "Docs Retrieved (Expanded)": len(ranked_expanded.get(qid, [])),
                "MAP Original": 0.0,
                "MAP Expanded": 0.0,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "Query ID",
            "Query",
            "Expansion Terms",
            "Docs Retrieved (Original)",
            "Docs Retrieved (Expanded)",
            "MAP Original",
            "MAP Expanded",
        ],
    )
    return {"num_queries": len(queries), "summary_df": summary_df}


# Backend adapters
def format_ranking_to_df(raw_ranking):
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

    _METHOD_OPTIONS = [
        "TF (logarithmic/binary/augmented/raw)",
        "IDF Only",
        "TF-IDF",
        "TF-IDF + Cosine Normalization",
    ]
    _TF_OPTIONS = ["raw tf", "binary tf", "log tf", "augmented tf"]
    _NORM_OPTIONS = ["no", "yes"]

    st.markdown("#### Document Weighting")
    doc_method = st.selectbox(
        "Document Weighting Method", options=_METHOD_OPTIONS, index=3, key="doc_method"
    )
    doc_tf_variant = st.selectbox(
        "Document TF Variant", options=_TF_OPTIONS, index=0, key="doc_tf_variant"
    )
    doc_norm = st.selectbox(
        "Document Normalization", options=_NORM_OPTIONS, index=1, key="doc_norm"
    )

    st.markdown("#### Query Weighting")
    query_method = st.selectbox(
        "Query Weighting Method", options=_METHOD_OPTIONS, index=3, key="query_method"
    )
    query_tf_variant = st.selectbox(
        "Query TF Variant", options=_TF_OPTIONS, index=0, key="query_tf_variant"
    )
    query_norm = st.selectbox(
        "Query Normalization", options=_NORM_OPTIONS, index=1, key="query_norm"
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

settings = {
    "use_stemming": use_stemming,
    "remove_stopwords": remove_stopwords,
    "add_all_words": add_all_words,
    "expansion_limit": int(expansion_limit),
    "doc_method": doc_method,
    "doc_tf_variant": doc_tf_variant,
    "doc_norm": doc_norm,
    "query_method": query_method,
    "query_tf_variant": query_tf_variant,
    "query_norm": query_norm,
}

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
    query = st.text_input(
        "Search query",
        placeholder="e.g. automatic information retrieval from titles",
        key="raw_query",
    )
    generate_clicked = st.button(
        "Generate Expansion Terms", type="primary", key="generate_btn"
    )

    if generate_clicked:
        if not query.strip():
            st.warning("Please enter a search query first.")
        else:
            with st.spinner("Generating expansion terms (GAN)..."):
                terms = run_expand_query(query, settings)
            st.session_state["expansion_terms"] = terms
            st.session_state["expansion_query"] = query
            all_term_strs = [t for t, _ in terms]
            if settings["add_all_words"]:
                st.session_state["selected_terms"] = all_term_strs
            else:
                st.session_state["selected_terms"] = all_term_strs[
                    : settings["expansion_limit"]
                ]
            st.session_state.pop("search_results", None)

    if "expansion_terms" in st.session_state:
        expansion_terms = st.session_state["expansion_terms"]
        option_strs = [t for t, _ in expansion_terms]

        st.subheader("Review Expansion Terms")
        if not option_strs:
            st.info("The GAN returned no expansion terms for this query.")
        st.multiselect(
            "Select the expansion terms to include in the search",
            options=option_strs,
            key="selected_terms",
        )
        execute_clicked = st.button("Execute Search", type="primary", key="execute_btn")

        if execute_clicked:
            selected = set(st.session_state.get("selected_terms", []))
            selected_pairs = [(t, w) for t, w in expansion_terms if t in selected]
            search_query = st.session_state.get("expansion_query", query)
            with st.spinner("Running retrieval..."):
                st.session_state["search_results"] = run_search(
                    search_query, settings, expanded_terms=selected_pairs
                )

    results = st.session_state.get("search_results")
    if results:
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

        # Document Viewer
        st.subheader("Document Viewer")
        retrieved_ids = list(
            dict.fromkeys(
                results["ranking_original"]["Document ID"].tolist()
                + results["ranking_expanded"]["Document ID"].tolist()
            )
        )
        if not retrieved_ids:
            st.info("No documents to view for this query.")
        else:
            viewer_system = get_retrieval_system(
                settings["use_stemming"],
                settings["remove_stopwords"],
                _WEIGHT_TO_ENUM[settings["doc_tf_variant"]],
                _NORM_TO_ENUM[settings["doc_norm"]],
            )
            raw_texts = viewer_system["raw_texts"]
            selected_doc = st.selectbox(
                "Select a retrieved Document ID to read its text",
                options=retrieved_ids,
                key="doc_viewer_select",
            )
            with st.expander(f"Document {selected_doc}", expanded=True):
                st.write(
                    raw_texts.get(
                        str(selected_doc),
                        "No text found for this document.",
                    )
                )


# Tab 2: Batch Processing
with batch_tab:
    st.subheader("Batch Processing")
    st.caption(
        "Upload a .txt file containing multiple queries (one query per line, or a "
        "CISI query.text file). Real GAN expansion and retrieval run for every "
        "query. (MAP is pending the qrels.text evaluation.)"
    )

    uploaded_file = st.file_uploader("Upload queries file", type=["text"])
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

            with st.spinner("Processing batch queries (GAN expansion + retrieval)..."):
                batch_result = run_batch_processing(uploaded_file, settings)
            st.success(f"Processed {batch_result['num_queries']} queries successfully.")

            st.dataframe(
                batch_result["summary_df"],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Batch Summary (CSV)",
                data=batch_result["summary_df"].to_csv(index=False),
                file_name="batch_summary.csv",
                mime="text/csv",
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
                _WEIGHT_TO_ENUM[settings["doc_tf_variant"]],
                _NORM_TO_ENUM[settings["doc_norm"]],
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