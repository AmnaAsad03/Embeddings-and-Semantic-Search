import json
from pathlib import Path

import chromadb
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
    / "chunks.json"
)

EMBEDDINGS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings"
    / "embeddings.npy"
)

VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"

COLLECTION_NAME = "medical_knowledge_base"


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def load_embeddings():
    return np.load(EMBEDDINGS_FILE)


def create_vector_database(chunks, embeddings):
    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Medical Knowledge Assistant vector database"
        }
    )

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    return collection


def main():
    print("=" * 70)
    print("MEDICAL VECTOR DATABASE")
    print("=" * 70)

    chunks = load_chunks()
    embeddings = load_embeddings()

    print(f"Chunks loaded : {len(chunks)}")
    print(f"Embeddings    : {embeddings.shape}")

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings do not match."
        )

    collection = create_vector_database(
        chunks,
        embeddings
    )

    print()
    print("=" * 70)
    print("VECTOR DATABASE COMPLETE")
    print("=" * 70)

    print(f"Collection    : {COLLECTION_NAME}")
    print(f"Documents     : {collection.count()}")
    print(f"Database      : {VECTOR_DB_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()