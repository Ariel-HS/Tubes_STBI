import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

_stemmer = PorterStemmer()
_stopwords = None

# Returns the cached English stopword set, downloading the NLTK corpus on first use
def _get_stopwords() -> set[str]:
    global _stopwords
    if _stopwords is None:
        try:
            _stopwords = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords")
            _stopwords = set(stopwords.words("english"))
    return _stopwords

# Parses the cisi.all or query.text file into { id: text }, concatenating the given fields.
# CISI documents (cisi.all) use .T (title) and .W (text); queries (query.text) use .W.
def load_smart(path: str, fields: tuple[str, ...] = ("T", "W")) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current_id = None
    current_field = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(".I"):
                current_id = line[2:].strip()
                current_field = None
                records[current_id] = []
            elif line.startswith("."):
                current_field = line[1:2]
            elif current_id is not None and current_field in fields:
                records[current_id].append(line.strip())

    return {doc_id: " ".join(parts) for doc_id, parts in records.items()}

# Tokenizes one string into a list of terms.
def preprocess(text: str, stem: bool = True, remove_stopwords: bool = True) -> list[str]:
    tokens = re.findall(r"[a-z]+", text.lower())

    if remove_stopwords:
        sw = _get_stopwords()
        tokens = [t for t in tokens if t not in sw]

    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]

    return tokens

# Applies preprocess() across a { id: text } collection.
def preprocess_collection(
    docs: dict[str, str], stem: bool = True, remove_stopwords: bool = True
) -> dict[str, list[str]]:
    return {
        doc_id: preprocess(text, stem=stem, remove_stopwords=remove_stopwords)
        for doc_id, text in docs.items()
    }
