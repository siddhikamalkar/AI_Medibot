# build_rag_database_from_pdf.py

import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

# Set paths
pdf_path = r"D:\medical-chatbot\data\The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"  # 🔵 Put correct PDF path
save_folder = r"C:\Users\DELL\Desktop\embeddings\db_faiss"  # 🔵 Folder where you want index.faiss and index.pkl

os.makedirs(save_folder, exist_ok=True)

# Load embedding model
print("🔵 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Read PDF and extract text
print("📄 Reading PDF...")
doc = fitz.open(pdf_path)
full_text = ""
for page in doc:
    full_text += page.get_text()

print(f"✅ PDF loaded. Total characters: {len(full_text)}")

# Split text into small chunks (around 500-700 characters)
print("✂️ Splitting into text chunks...")
chunk_size = 700
overlap = 100

def split_text(text, chunk_size=700, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

passages = split_text(full_text, chunk_size=chunk_size, overlap=overlap)

print(f"✅ Total chunks created: {len(passages)}")

# Generate embeddings
print("🔵 Generating embeddings...")
embeddings = embedder.encode(passages, batch_size=32, show_progress_bar=True)

# Create FAISS index
print("🛠 Creating FAISS index...")
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(embeddings).astype(np.float32))

# Save FAISS index
faiss.write_index(faiss_index, os.path.join(save_folder, "index.faiss"))

# Save passages
with open(os.path.join(save_folder, "index.pkl"), "wb") as f:
    pickle.dump(passages, f)

print("✅ Successfully saved FAISS index and passages!")

print("\n🏁 DONE! Now you can load these files in your chatbot.")
