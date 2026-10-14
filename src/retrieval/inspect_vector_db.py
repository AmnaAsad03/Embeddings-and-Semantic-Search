import chromadb
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"

COLLECTION_NAME = "medical_knowledge_base"


def main():
    print("=" * 70)
    print("CHROMADB INSPECTION")
    print("=" * 70)

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(f"Collection     : {COLLECTION_NAME}")
    print(f"Total records  : {collection.count()}")

    print()
    print("-" * 70)
    print("SAMPLE RECORDS")
    print("-" * 70)

    results = collection.get(
        limit=3,
        include=[
            "documents",
            "metadatas",
            "embeddings"
        ]
    )

    for i in range(len(results["ids"])):

        print()
        print(f"Record {i + 1}")
        print("-" * 50)

        print(f"ID:")
        print(results["ids"][i])

        print()
        print("TEXT:")
        print(results["documents"][i])

        print()
        print("METADATA:")
        for key, value in results["metadatas"][i].items():
            print(f"  {key}: {value}")

        print()
        print("EMBEDDING:")
        embedding = results["embeddings"][i]

        print(f"  Dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()