import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# free local embedding model, no API key needed
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# load merged chunks from orphanet + hpo
with open("../data/chunks_merged.json", "r") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]
print(f"embedding {len(texts)} chunks...")

# embed all texts locally
def embed_texts(texts, batch_size=100):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)
        print(f"  embedded {min(i+batch_size, len(texts))}/{len(texts)}")
    return np.array(all_embeddings, dtype="float32")

embeddings = embed_texts(texts)

# build faiss index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print(f"faiss index built with {index.ntotal} vectors")

# save index and chunks for step 3
faiss.write_index(index, "../data/merged_index.faiss")
with open("../data/chunks_merged.json", "w") as f:
    json.dump(chunks, f)
print("saved merged_index.faiss and chunks_merged.json")