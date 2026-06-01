from preprocessing.preprocessing import load_smart, preprocess_collection

CISI_PATH = "data/cisi.all"
QUERY_PATH = "data/query.text"

# Load raw documents
documents = load_smart(CISI_PATH)

# Preprocess
processed_docs = preprocess_collection(documents)

# Show the result for a few documents
print(f"Loaded and preprocessed {len(processed_docs)} documents\n")
for doc_id in list(processed_docs)[:5]:
    print(f"Document {doc_id}: {processed_docs[doc_id]}")

# Load raw query
documents = load_smart(QUERY_PATH)

# Preprocess
processed_docs = preprocess_collection(documents)

# Show the result for a few documents
print(f"Loaded and preprocessed {len(processed_docs)} queries\n")
for doc_id in list(processed_docs)[:5]:
    print(f"Document {doc_id}: {processed_docs[doc_id]}")
