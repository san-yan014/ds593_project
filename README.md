# Rare Disease Diagnosis Assistant

A retrieval-augmented generation (RAG) system that helps physicians identify rare disease candidates from patient symptom profiles. The system grounds LLM reasoning in structured rare disease databases, reducing hallucination risk in a safety-critical clinical setting.

> ⚕️ **This tool is intended for clinical use only. It is not a substitute for professional medical judgment.**

---

## Overview

Rare diseases affect 1 in 10 Americans but take an average of 4–7 years to diagnose. This tool takes a list of clinical symptoms as input and returns ranked rare disease candidates with symptom-by-symptom reasoning and suggested next steps — all grounded in real medical databases.

---

## Features

- Retrieves the most relevant rare diseases from **16,905 disease profiles** across HPO and Orphanet
- Uses Claude (`claude-sonnet-4-6`) to rank and explain matches with chain-of-thought reasoning
- Blocks non-clinical self-diagnosis attempts via prompt guardrails
- Flags low-confidence retrievals instead of returning unreliable results
- Clean Streamlit UI for interactive clinical use

---

## Data Sources

| Source | Description | Size |
|--------|-------------|------|
| [Human Phenotype Ontology (HPO)](https://hpo.jax.org) | Standardized symptom vocabulary + disease-phenotype annotations | 12,570 diseases |
| [Orphanet](https://www.orphadata.com/_phenotypes/) | Rare disease registry with HPO-aligned symptom mappings | 4,335 diseases |

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Embeddings | `BAAI/bge-large-en-v1.5` via sentence-transformers (local, free) |
| Vector store | FAISS (IndexFlatL2) |
| LLM | Claude `claude-sonnet-4-6` via Anthropic API |
| UI | Streamlit |
| Language | Python 3.13 |

---

## Project Structure

```
ds593_project/
├── code/
│   ├── parse_data.py        # parses hp.obo and phenotype.hpoa
│   ├── orphanet_parser.py   # parses en_product4.xml and merges with HPO
│   ├── embed_chunks.py      # embeds chunks and builds FAISS index
│   ├── retrieval.py         # basic retrieval loop
│   ├── reasoning.py         # full RAG pipeline with Claude + guardrails
│   └── streamlit_app.py     # streamlit UI
├── data/                    # not tracked in git (see .gitignore)
│   ├── hp.obo
│   ├── phenotype.hpoa
│   ├── en_product4.xml
│   ├── chunks_merged.json
│   └── merged_index.faiss
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/san-yan014/ds593_project
cd ds593_project
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download data files
Place the following in the `data/` folder:
- `hp.obo` and `phenotype.hpoa` from [hpo.jax.org](https://hpo.jax.org/data/ontology)
- `en_product4.xml` (English) from [orphadata.com](https://www.orphadata.com/_phenotypes/)

### 5. Set your Anthropic API key
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### 6. Build the knowledge base
```bash
cd code
python parse_data.py
python orphanet_parser.py
python embed_chunks.py
```

### 7. Run the app
```bash
streamlit run streamlit_app.py
```

---

## Usage

Enter clinical symptoms using medical terminology:

```
ptosis, dysphagia, proximal muscle weakness
fatigue, joint pain, skin rash
seizures, intellectual disability, abnormal gait
```

The system returns ranked disease candidates with matched and missing symptoms, unaccounted symptoms, and suggested clinical next steps.

---

## Safety

- Patient self-diagnosis queries are blocked via keyword guardrails
- Low-confidence retrievals are flagged rather than returned
- All outputs include a disclaimer to confirm findings with appropriate testing

---

## Requirements

```
anthropic
sentence-transformers
faiss-cpu
numpy
streamlit
```

---

## Course

DS593 — Natural Language Processing  
Final Project — Spring 2025
