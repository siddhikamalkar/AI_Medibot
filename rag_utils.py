import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the same model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS and Passages
faiss_index = faiss.read_index(r"C:\Users\DELL\Desktop\embeddings\db_faiss\index.faiss")
with open(r"C:\Users\DELL\Desktop\embeddings\db_faiss\index.pkl", "rb") as f:
    passages = pickle.load(f)

# Retrieval function
def retrieve_context(user_query, top_k=3):
    if not user_query.strip():
        return "No additional medical context found."

    query_embedding = embedder.encode([user_query])
    distances, indices = faiss_index.search(np.array(query_embedding).astype(np.float32), top_k)

    if indices is None or len(indices) == 0 or len(indices[0]) == 0:
        return "No relevant medical knowledge found."

    retrieved_passages = [passages[idx] for idx in indices[0] if idx < len(passages)]
    combined_context = "\n\n".join(retrieved_passages)
    return combined_context
