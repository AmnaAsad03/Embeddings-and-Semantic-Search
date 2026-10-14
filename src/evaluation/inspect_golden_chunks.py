from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
    / "chunks.json"
)


SEARCH_TERMS = {
    "MED-001": [
        "symptoms",
        "wheezing",
        "shortness of breath",
        "asthma attack",
    ],
    "MED-002": [
        "symptoms",
        "thirst",
        "urination",
        "blood sugar",
    ],
    "MED-003": [
        "symptoms",
        "causes",
        "shortness of breath",
        "pulmonary hypertension",
    ],
    "MED-004": [
        "symptoms",
        "causes",
        "acid",
        "reflux",
    ],
    "MED-005": [
        "symptoms",
        "headache",
        "nausea",
        "migraine",
    ],
    "MED-006": [
        "symptoms",
        "causes",
        "iron",
        "anemia",
    ],
    "MED-007": [
        "cardiovascular",
        "risk factors",
        "prevention",
        "heart",
    ],
    "MED-008": [
        "symptoms",
        "causes",
        "prevention",
        "pneumonia",
    ],
    "MED-009": [
        "symptoms",
        "prevention",
        "vaccine",
        "influenza",
    ],
    "MED-010": [
        "symptoms",
        "causes",
        "bacteria",
        "food poisoning",
    ],
    "MED-011": [
        "symptoms",
        "treatment",
        "depression",
        "help",
    ],
    "MED-012": [
        "safety",
        "warning",
        "medication",
        "drug",
    ],
}


def load_chunks():
    print("Loading chunks...")

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    print(f"Chunks loaded: {len(chunks)}")

    return chunks


def inspect_document(chunks, document_id, terms):
    print("\n" + "=" * 80)
    print(f"{document_id}")
    print("=" * 80)

    document_chunks = [
        chunk
        for chunk in chunks
        if chunk["metadata"].get("document_id") == document_id
    ]

    print(f"Total chunks: {len(document_chunks)}")

    found_chunks = []

    for chunk in document_chunks:

        text = chunk["text"].lower()

        matched_terms = [
            term
            for term in terms
            if term.lower() in text
        ]

        if matched_terms:

            found_chunks.append(
                (chunk, matched_terms)
            )

    for chunk, matched_terms in found_chunks:

        print("\n" + "-" * 80)

        print(
            f"Chunk ID       : "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Chunk Index    : "
            f"{chunk['chunk_index']}"
        )

        print(
            f"Matched Terms  : "
            f"{matched_terms}"
        )

        preview = (
            chunk["text"]
            .replace("\n", " ")
            .strip()
        )

        print("\nContent:")
        print(preview[:700])

    print(
        f"\nCandidate chunks found: "
        f"{len(found_chunks)}"
    )


def main():

    print("=" * 80)
    print("GOLDEN RETRIEVAL CHUNK INSPECTION")
    print("=" * 80)

    chunks = load_chunks()

    for document_id, terms in SEARCH_TERMS.items():

        inspect_document(
            chunks,
            document_id,
            terms,
        )

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)

    print(
        "\nUse the output to manually identify "
        "answer-bearing chunk IDs for the golden dataset."
    )


if __name__ == "__main__":
    main()