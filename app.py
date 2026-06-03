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

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        body { background-color: #ebebeb; color: #334155; }
    </style>
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
