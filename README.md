# Tubes STBI — Information Retrieval Engine

A Vector Space Model Information Retrieval engine for the **IF4042 Sistem Temu Balik Informasi** course project. It ranks CISI documents against a query using **TF-IDF weighting** and **cosine similarity**, with an optional **GAN-based query expander** that widens the query before retrieval.

A [Streamlit](https://streamlit.io/) app (`app.py`) is the main user-facing front end.

## Group Member

- Ariel Herfrison 13522002
- Ibrahim Ihsan Rasyid 13522018
- Farhan Nafis Rayhan 13522037
- Fabian Radenta Bangun 13522105
- Rayendra Althaf Taraka Noor 13522107



## Features

- TF-IDF retrieval with cosine similarity over the CISI collection
- Configurable weighting: Raw / Log / Binary / Augmented TF, with optional cosine normalization
- GAN-based query expansion
- Three UI tabs: **Interactive Search**, **Batch Processing**, and **Inverted Index Inspector**

## Requirements

- **Python 3.9+** (uses built-in generic syntax like `dict[str, list[str]]` and implicit namespace packages)
- Dependencies listed in `requirements.txt`: `streamlit`, `pandas`, `nltk`, `numpy`, `scipy`, `scikit-learn`, `torch` (plus `matplotlib` and `jupyter`, used only by the benchmark notebook)

## Installation

```bash
# (optional) create a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

> The NLTK stopwords corpus is downloaded automatically on first run (requires an internet connection).

## Usage

> All scripts must be run from the repository root so package-style imports resolve.

Run the Streamlit app (main entry point):

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Benchmark

`benchmark.py` evaluates retrieval quality across **all 32 scoring variations** (4 preprocessing × 4 TF weights × 2 normalization schemes), comparing **MAP** for original vs GAN-expanded queries over the judged queries in `qrels.text`.

```bash
python benchmark.py # prints the numerical result table and writes benchmark_results.csv
```

For a color-graded table view, open `benchmark.ipynb` (VS Code or Jupyter) and Run All — it reuses `run_benchmark()` from `benchmark.py`.

## Architecture

| Module | Responsibility |
|---|---|
| `document_retrieval/retrieval.py` | Score and rank documents by cosine similarity |
| `inverted_index/inverted_index.py` | Build `{term: {doc_id: raw_tf}}` from tokenized docs |
| `preprocessing/preprocessing.py` | Parse SMART-format files, tokenize, remove stopwords, Porter-stem |
| `query_expansion/query_expansion.py` | GAN-based query expander |
| `scoring/scoring.py` | Compute IDF, TF-IDF weights, and document Euclidean lengths |
| `app.py` | Streamlit front end |

## Test Collection

The `data/` directory holds the **CISI** dataset in SMART format:

- `cisi.all` — documents (`.I` id, `.T` title, `.A` author, `.W` text, `.X` cross-references)
- `query.text` — queries (`.I` id, `.W` text)
- `qrels.text` — relevance judgments for evaluation
