import json
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

# load model, index, and chunks
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
index = faiss.read_index("../data/merged_index.faiss")
client = anthropic.Anthropic(api_key="your-key-here")

with open("../data/chunks_merged.json", "r") as f:
    chunks = json.load(f)

# retrieve top-k diseases for a given symptom query
def retrieve(query, k=5):
    query_vec = model.encode([query], show_progress_bar=False).astype("float32")
    distances, indices = index.search(query_vec, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "disease": chunks[idx]["disease"],
            "symptoms": chunks[idx]["symptoms"],
            "score": round(float(dist), 4)
        })
    return results

# build prompt from retrieved results
def build_prompt(query, results):
    candidates = ""
    for i, r in enumerate(results):
        candidates += f"\n{i+1}. {r['disease']}\n   symptoms: {', '.join(r['symptoms'][:8])}\n"

    return f"""You are a clinical decision support assistant helping physicians investigate rare disease diagnoses.

A patient presents with the following symptoms: {query}

Here are the top candidate diseases retrieved from a rare disease database:
{candidates}
Your task:
1. Rank these diseases from most to least likely given the symptoms
2. For each disease, explain which symptoms match and which are missing
3. Flag any symptoms that are unaccounted for by all candidates
4. End with a brief clinical note on what to investigate next

Be concise, factual, and always remind the physician to confirm with further testing."""

# safety: block non-clinical queries
NON_CLINICAL_TRIGGERS = [
    "i have", "i am", "i feel", "my symptoms", "do i have",
    "am i sick", "what's wrong with me", "i think i have"
]

def is_non_clinical(query):
    q = query.lower()
    return any(trigger in q for trigger in NON_CLINICAL_TRIGGERS)

# safety: check if retrieval confidence is strong enough
CONFIDENCE_THRESHOLD = 0.8  # l2 distance — above this means weak match

def check_confidence(results):
    best_score = results[0]["score"]
    return best_score < CONFIDENCE_THRESHOLD

# full rag pipeline: retrieve + reason
def diagnose(query, k=5):
    # guard: block patient self-diagnosis attempts
    if is_non_clinical(query):
        return ("⚠️ This tool is intended for clinical use only. "
                "Please consult a licensed physician for personal medical concerns.")

    results = retrieve(query, k)

    # guard: flag weak retrieval confidence
    if not check_confidence(results):
        return (f"⚠️ Low retrieval confidence (best score: {results[0]['score']:.4f}). "
                "The symptoms provided may be too vague or uncommon for reliable matching. "
                "Please provide more specific clinical terminology and try again.")

    prompt = build_prompt(query, results)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# interactive loop
if __name__ == "__main__":
    print("rare disease diagnosis assistant — type symptoms to diagnose, 'quit' to exit\n")
    while True:
        query = input("symptoms: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue
        print("\nanalyzing...\n")
        print(diagnose(query))
        print("\n" + "─"*60 + "\n")