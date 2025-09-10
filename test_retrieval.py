# test_retrieval.py

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and passages
faiss_index = faiss.read_index(r"C:\Users\DELL\Desktop\embeddings\db_faiss\index.faiss")
with open(r"C:\Users\DELL\Desktop\embeddings\db_faiss\index.pkl", "rb") as f:
    passages = pickle.load(f)

# Simple function to retrieve relevant context
def retrieve_context(user_query, top_k=3):
    if not user_query.strip():
        return "Please enter a valid query."

    query_embedding = embedder.encode([user_query])
    distances, indices = faiss_index.search(np.array(query_embedding).astype(np.float32), top_k)

    if indices is None or len(indices) == 0 or len(indices[0]) == 0:
        return "No relevant medical knowledge found."

    retrieved_passages = [passages[idx] for idx in indices[0] if idx < len(passages)]
    combined_context = "\n\n".join(retrieved_passages)
    return combined_context

# ---------------------
# Test code
# ---------------------

print("\n🩺 Welcome to MediBot RAG Tester 🩺")

while True:
    user_query = input("\n🔹 Enter a medical question (or type 'exit' to quit):\n> ")

    if user_query.lower() == "exit":
        print("\n👋 Exiting MediBot RAG Tester. Goodbye!")
        break

    retrieved = retrieve_context(user_query)
    print("\n🔎 Retrieved Medical Knowledge:\n")
    print(retrieved)
    print("\n" + "-"*60)
