import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# load model, index, and chunks
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
index = faiss.read_index("hpo_index.faiss")

with open("chunks.json", "r") as f:
    chunks = json.load(f)

# retrieve top-k diseases for a given symptom query
def retrieve(query, k=5):
    query_vec = model.encode([query], show_progress_bar=False).astype("float32")
    distances, indices = index.search(query_vec, k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        results.append({
            "rank": rank + 1,
            "disease": chunks[idx]["disease"],
            "symptoms": chunks[idx]["symptoms"],
            "score": round(float(dist), 4)
        })
    return results

# simple interactive loop
if __name__ == "__main__":
    print("rare disease retrieval — type symptoms to search, 'quit' to exit\n")
    while True:
        query = input("symptoms: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        results = retrieve(query)
        print(f"\ntop {len(results)} matches:")
        for r in results:
            print(f"\n  #{r['rank']} {r['disease']} (score: {r['score']})")
            print(f"  symptoms: {', '.join(r['symptoms'][:5])}...")
        print()