import json
import xml.etree.ElementTree as ET

# parse en_product4.xml into disease chunks
def parse_orphanet(path):
    tree = ET.parse(path)
    root = tree.getroot()
    chunks = []

    for disorder in root.iter("Disorder"):
        # get disease name
        name_el = disorder.find("Name")
        if name_el is None:
            continue
        disease_name = name_el.text.strip()

        # get associated HPO symptoms
        symptoms = []
        for assoc in disorder.iter("HPODisorderAssociation"):
            hpo_term = assoc.find(".//HPOTerm")
            if hpo_term is not None:
                symptoms.append(hpo_term.text.strip())

        if not symptoms:
            continue

        chunks.append({
            "disease": disease_name,
            "symptoms": symptoms,
            "text": f"Disease: {disease_name}\nSymptoms: {', '.join(symptoms)}",
            "source": "orphanet"
        })

    return chunks

# merge orphanet chunks into existing hpo chunks
def merge_chunks(hpo_path, orphanet_chunks, output_path):
    with open(hpo_path, "r") as f:
        hpo_chunks = json.load(f)

    # tag existing hpo chunks with source
    for c in hpo_chunks:
        c["source"] = "hpo"

    merged = hpo_chunks + orphanet_chunks
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    return merged

if __name__ == "__main__":
    print("parsing en_product4.xml...")
    orphanet_chunks = parse_orphanet("../data/en_product4.xml")
    print(f"  loaded {len(orphanet_chunks)} orphanet diseases")

    print("merging with hpo chunks...")
    merged = merge_chunks("../data/chunks.json", orphanet_chunks, "../data/chunks_merged.json")
    print(f"  total chunks: {len(merged)}")

    print("\nsample orphanet chunk:")
    print(orphanet_chunks[0]["text"][:300])