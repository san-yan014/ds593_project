import os
import streamlit as st
import json
import numpy as np
import faiss
import anthropic
from sentence_transformers import SentenceTransformer

# ── page config
st.set_page_config(page_title="Rare Disease Diagnosis Assistant", page_icon="🔬", layout="centered")

# ── load model, index, chunks once
@st.cache_resource
def load_resources():
    m = SentenceTransformer("BAAI/bge-large-en-v1.5")
    idx = faiss.read_index("../data/merged_index.faiss")
    with open("../data/chunks_merged.json", "r") as f:
        ch = json.load(f)
    return m, idx, ch

model, index, chunks = load_resources()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── safety
NON_CLINICAL_TRIGGERS = [
    "i have", "i am", "i feel", "my symptoms", "do i have",
    "am i sick", "what's wrong with me", "i think i have"
]
CONFIDENCE_THRESHOLD = 0.8

def is_non_clinical(query):
    return any(t in query.lower() for t in NON_CLINICAL_TRIGGERS)

# ── retrieval
def retrieve(query, k=5):
    vec = model.encode([query], show_progress_bar=False).astype("float32")
    distances, indices = index.search(vec, k)
    return [{"disease": chunks[i]["disease"], "symptoms": chunks[i]["symptoms"], "score": round(float(d), 4)}
            for d, i in zip(distances[0], indices[0])]

# ── prompt
def build_prompt(query, results):
    candidates = "".join(
        f"\n{i+1}. {r['disease']}\n   symptoms: {', '.join(r['symptoms'][:8])}\n"
        for i, r in enumerate(results)
    )
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

# ── diagnose
def diagnose(query):
    if is_non_clinical(query):
        return "⚠️ This tool is intended for clinical use only. Please consult a licensed physician for personal medical concerns."
    results = retrieve(query)
    if results[0]["score"] >= CONFIDENCE_THRESHOLD:
        return f"⚠️ Low retrieval confidence (best score: {results[0]['score']:.4f}). Please provide more specific clinical terminology."
    prompt = build_prompt(query, results)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# ── ui
st.title("🔬 Rare Disease Diagnosis Assistant")
st.caption("A clinical decision support tool for physicians. Not intended for patient self-diagnosis.")
st.warning("⚕️ For clinical use only. Always confirm findings with appropriate testing and specialist consultation.")

st.markdown("### Enter Patient Symptoms")
query = st.text_area("Describe symptoms in clinical terms (e.g. 'ptosis, dysphagia, proximal muscle weakness')", height=100)

if st.button("Analyze", type="primary"):
    if not query.strip():
        st.error("Please enter at least one symptom.")
    else:
        with st.spinner("Retrieving and analyzing..."):
            result = diagnose(query)
        st.markdown("---")
        st.markdown("### Clinical Analysis")
        st.markdown(result)