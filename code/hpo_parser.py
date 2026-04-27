import json
import re
from collections import defaultdict

# parse hp.obo into {HPO_ID: {"name": ..., "synonyms": [...]}}
def parse_obo(path):
    terms = {}
    cur = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                if cur.get("id", "").startswith("HP:"):
                    terms[cur["id"]] = {
                        "name": cur.get("name", ""),
                        "synonyms": cur.get("synonyms", [])
                    }
                cur = {"synonyms": []}
            elif line.startswith("id: "):
                cur["id"] = line[4:]
            elif line.startswith("name: "):
                cur["name"] = line[6:]
            elif line.startswith("synonym: "):
                m = re.search(r'"(.+?)"', line)
                if m:
                    cur["synonyms"].append(m.group(1))
    return terms

# parse phenotype.hpoa into {disease_name: [hpo_ids]}
def parse_hpoa(path):
    diseases = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or line.startswith("database_id"):
                continue
            cols = line.strip().split("\t")
            if len(cols) < 4:
                continue
            disease_name = cols[1]
            hpo_id = cols[3]
            qualifier = cols[2]
            if qualifier != "NOT" and hpo_id.startswith("HP:"):
                diseases[disease_name].append(hpo_id)
    return dict(diseases)

# build one text chunk per disease for embedding
def build_chunks(diseases, hpo_terms):
    chunks = []
    for disease, hpo_ids in diseases.items():
        symptoms = [hpo_terms[hid]["name"] for hid in hpo_ids if hid in hpo_terms]
        if not symptoms:
            continue
        chunks.append({
            "disease": disease,
            "symptoms": symptoms,
            "text": f"Disease: {disease}\nSymptoms: {', '.join(symptoms)}"
        })
    return chunks

if __name__ == "__main__":
    OBO_PATH = "hp.obo"
    HPOA_PATH = "phenotype.hpoa"

    print("parsing hp.obo...")
    hpo_terms = parse_obo(OBO_PATH)
    print(f"  loaded {len(hpo_terms)} HPO terms")

    print("parsing phenotype.hpoa...")
    diseases = parse_hpoa(HPOA_PATH)
    print(f"  loaded {len(diseases)} diseases")

    print("building chunks...")
    chunks = build_chunks(diseases, hpo_terms)
    print(f"  built {len(chunks)} chunks")

    with open("chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)
    print("  saved to chunks.json")

    print("\nsample chunk:")
    print(chunks[0]["text"][:300])