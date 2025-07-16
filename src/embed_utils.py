
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import pickle
import os

# Load cleaned dataset
df = pd.read_csv(r"C:\Users\Administrator\creditrust-rag-chatbot\data\processed\filtered_complaints.csv")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", "!", "?", ",", " "]
)

# Prepare data for chunking
chunks = []
metadata = []

for idx, row in df.iterrows():
    complaint_id = row.get('Complaint ID', idx)
    product = row['Product']
    text = row['cleaned_narrative']
    split_texts = text_splitter.split_text(text)
    for chunk in split_texts:
        chunks.append(chunk)
        metadata.append({'complaint_id': complaint_id, 'product': product})

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(chunks, show_progress_bar=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Persist index and metadata
os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, "vector_store/faiss_index.bin")

with open("vector_store/chunk_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ FAISS index and metadata saved in vector_store/")
