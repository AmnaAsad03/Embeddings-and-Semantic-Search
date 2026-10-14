import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
    / "chunks.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings"
)

OUTPUT_FILE = OUTPUT_DIR / "embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def create_embeddings(chunks):
    model = SentenceTransformer(MODEL_NAME)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return np.array(embeddings)


def main():
    print("=" * 70)
    print("MEDICAL DOCUMENT EMBEDDING")
    print("=" * 70)

    chunks = load_chunks()

    print(f"Chunks loaded : {len(chunks)}")
    print(f"Model         : {MODEL_NAME}")

    embeddings = create_embeddings(chunks)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        OUTPUT_FILE,
        embeddings
    )

    print()
    print("=" * 70)
    print("EMBEDDING COMPLETE")
    print("=" * 70)

    print(f"Chunks        : {len(chunks)}")
    print(f"Embeddings    : {embeddings.shape}")
    print(f"Dimension     : {embeddings.shape[1]}")
    print(f"Output        : {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()