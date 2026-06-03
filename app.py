import random

import pandas as pd
import streamlit as st


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


# ---------------------------------------------------------------------------
# Mock (dummy CISI-format data)
# ---------------------------------------------------------------------------
# These imitate the shape of data the real backend will return. Replace them
# with the real engine (GAN query expansion, MAP evaluation, retrieval) later.
EXPANSION_CANDIDATES = [
    "retrieval", "document", "ranking", "relevance", "semantic",
    "indexing", "similarity", "vector", "corpus", "feedback",
    "embedding", "synonym", "context", "language", "generative",
]

INDEX_TERMS = [
    "dewey", "classification", "library", "system", "edition",
    "index", "history", "decimal", "growth", "study",
    "automatic", "retrieval", "title", "article", "relevance",
]


def mock_expand_query(query, limit, add_all):
    """Return (term, weight) pairs imitating GAN query expansion."""
    random.seed(hash(query) & 0xFFFFFFFF)
    weights = [(t, round(random.uniform(0.30, 0.95), 3)) for t in EXPANSION_CANDIDATES]
    weights.sort(key=lambda x: x[1], reverse=True)
    if not add_all:
        weights = weights[: max(0, limit)]
    return weights


def mock_ranking(seed_key):
    """Return a dummy ranked-document table in CISI doc-id format."""
    random.seed(hash(seed_key) & 0xFFFFFFFF)
    rows = []
    score = 0.95
    for rank in range(1, 11):
        rows.append(
            {
                "Rank": rank,
                "Document ID": str(random.randint(1, 1488)),
                "Similarity": round(score, 4),
            }
        )
        score -= random.uniform(0.02, 0.08)
    return pd.DataFrame(rows)


def mock_search(query, settings):
    """Return mock retrieval results for a single query."""
    random.seed(hash(query) & 0xFFFFFFFF)
    expanded = mock_expand_query(
        query, settings["expansion_limit"], settings["add_all_words"]
    )
    return {
        "original_query": query,
        "expanded_terms": expanded,
        "map_original": round(random.uniform(0.20, 0.45), 4),
        "map_expanded": round(random.uniform(0.45, 0.70), 4),
        "ranking_original": mock_ranking(query + "_orig"),
        "ranking_expanded": mock_ranking(query + "_exp"),
    }


def mock_inverted_index(doc_id):
    """Return a wide mock inverted-index slice for one document."""
    random.seed(hash(doc_id) & 0xFFFFFFFF)
    rows = []
    for term in INDEX_TERMS:
        tf = random.randint(1, 12)
        rows.append(
            {
                "Term": term,
                "Term Frequency": tf,
                "Document Frequency": random.randint(1, 400),
                "IDF": round(random.uniform(0.5, 9.5), 4),
                "TF-IDF Weight": round(tf * random.uniform(0.5, 9.5), 4),
            }
        )
    return pd.DataFrame(rows)


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
        highlights = " ".join(
            f'<span class="highlight">{term}</span>' for term, _ in expanded_terms
        )
        body = f"{query} {highlights}"
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

settings = {
    "use_stemming": use_stemming,
    "remove_stopwords": remove_stopwords,
    "weighting_method": weighting_method,
    "tf_variant": tf_variant,
    "add_all_words": add_all_words,
    "expansion_limit": int(expansion_limit),
}


# Main area header
st.markdown('<div class="app-title">STBI Search Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Generative Adversarial Network Group</div>',
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
            results = mock_search(query, settings)

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
            batch_result = mock_process_batch(uploaded_file.getvalue())
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
            render_query_badge("Document", inspect_doc_id)
            st.dataframe(
                mock_inverted_index(inspect_doc_id),
                use_container_width=True,
                hide_index=True,
            )