from preprocessing.preprocessing import load_smart, preprocess_collection
from document_retrieval.retrieval import retrieve_documents

DATA = "data"

# 1. Load and preprocess documents and queries
docs = preprocess_collection(load_smart(f"{DATA}/cisi.all"))
queries = preprocess_collection(load_smart(f"{DATA}/query.text", fields=("W",)))

# 2. Retrieve and rank documents for every query
results = retrieve_documents(queries, docs)

# 3. Show the top 5 documents for query 1
print(f"{len(docs)} documents, {len(queries)} queries\n")
print("Query 1 - top 5 documents (doc_id, score):")
for doc_id, score in results["1"][:5]:
    print(f"  {doc_id}: {score:.4f}")
