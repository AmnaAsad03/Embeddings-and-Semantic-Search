from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_PATH = PROJECT_ROOT / "data" / "vector_db"
COLLECTION_NAME = "medical_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 5


class VectorSearcher:

    def __init__(
        self,
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        model_name=EMBEDDING_MODEL,
    ):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        print("Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=str(chroma_path)
        )

        self.collection = self.client.get_collection(
            name=collection_name
        )

        print("Vector search initialized.")
        print(f"Collection : {collection_name}")
        print(f"Documents  : {self.collection.count()}")

    def search(
        self,
        query,
        top_k=DEFAULT_TOP_K,
        filters=None,
    ):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        search_parameters = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if filters:
            search_parameters["where"] = self._build_where_clause(
                filters
            )

        results = self.collection.query(
            **search_parameters
        )

        retrieved_results = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for rank, (document, metadata, distance) in enumerate(
            zip(documents, metadatas, distances),
            start=1,
        ):
            retrieved_results.append({
                "rank": rank,
                "content": document,
                "metadata": metadata,
                "distance": distance,
            })

        return retrieved_results

    def _build_where_clause(self, filters):
        if not filters:
            return None

        conditions = []

        for field, value in filters.items():
            if value is None:
                continue

            conditions.append({
                field: {
                    "$eq": value
                }
            })

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {
            "$and": conditions
        }


if __name__ == "__main__":

    print("=" * 70)
    print("MEDICAL VECTOR SEARCH WITH METADATA FILTERING")
    print("=" * 70)

    searcher = VectorSearcher()

    query = "What are common symptoms?"

    print(f"\nQuery: {query}")

    print("\nFilter: topic = Respiratory")
    print("-" * 70)

    results = searcher.search(
        query=query,
        top_k=5,
        filters={
            "topic": "Respiratory"
        },
    )

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Distance: {result['distance']:.4f}")
        print(
            f"Document ID: "
            f"{result['metadata'].get('document_id')}"
        )
        print(
            f"Title: "
            f"{result['metadata'].get('title')}"
        )
        print(
            f"Topic: "
            f"{result['metadata'].get('topic')}"
        )
        print(
            f"Source: "
            f"{result['metadata'].get('source')}"
        )
        print("\nContent:")
        print(result["content"][:300])
        print("-" * 70)