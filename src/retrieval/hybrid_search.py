from pathlib import Path

from src.retrieval.vector_search import VectorSearcher
from src.retrieval.keyword_search import BM25Searcher


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 5

BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5


# ============================================================
# HYBRID SEARCHER
# ============================================================

class HybridSearcher:

    def __init__(self):

        print("Initializing hybrid search...")

        self.vector_searcher = VectorSearcher()
        self.bm25_searcher = BM25Searcher()

        print("Hybrid search initialized.")
        print(f"BM25 Weight   : {BM25_WEIGHT}")
        print(f"Vector Weight : {VECTOR_WEIGHT}")

    # --------------------------------------------------------
    # MIN-MAX NORMALIZATION
    # --------------------------------------------------------

    def _normalize(self, scores):

        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)

        if maximum == minimum:
            return [1.0] * len(scores)

        return [
            (score - minimum) / (maximum - minimum)
            for score in scores
        ]

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    def search(self, query, top_k=DEFAULT_TOP_K):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        vector_results = self.vector_searcher.search(
            query=query,
            top_k=top_k
        )

        bm25_results = self.bm25_searcher.search(
            query=query,
            top_k=top_k
        )

        # Vector distance: lower is better
        vector_distances = [
            result["distance"]
            for result in vector_results
        ]

        vector_scores = [
            1 / (1 + distance)
            for distance in vector_distances
        ]

        normalized_vector = self._normalize(
            vector_scores
        )

        # BM25 score: higher is better
        bm25_scores = [
            result["score"]
            for result in bm25_results
        ]

        normalized_bm25 = self._normalize(
            bm25_scores
        )

        combined_results = {}

        # ----------------------------------------------------
        # ADD VECTOR RESULTS
        # ----------------------------------------------------

        for result, normalized_score in zip(
            vector_results,
            normalized_vector
        ):

            chunk_id = result["metadata"].get(
                "chunk_id"
            )

            if not chunk_id:
                chunk_id = result["content"]

            combined_results[chunk_id] = {
                "content": result["content"],
                "metadata": result["metadata"],
                "vector_score": normalized_score,
                "bm25_score": 0.0,
            }

        # ----------------------------------------------------
        # ADD BM25 RESULTS
        # ----------------------------------------------------

        for result, normalized_score in zip(
            bm25_results,
            normalized_bm25
        ):

            chunk_id = result["metadata"].get(
                "chunk_id"
            )

            if not chunk_id:
                chunk_id = result["content"]

            if chunk_id not in combined_results:

                combined_results[chunk_id] = {
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "vector_score": 0.0,
                    "bm25_score": normalized_score,
                }

            else:

                combined_results[chunk_id][
                    "bm25_score"
                ] = normalized_score

        # ----------------------------------------------------
        # CALCULATE HYBRID SCORE
        # ----------------------------------------------------

        for result in combined_results.values():

            result["hybrid_score"] = (
                VECTOR_WEIGHT
                * result["vector_score"]
                +
                BM25_WEIGHT
                * result["bm25_score"]
            )

        # ----------------------------------------------------
        # SORT RESULTS
        # ----------------------------------------------------

        ranked_results = sorted(
            combined_results.values(),
            key=lambda result: result["hybrid_score"],
            reverse=True
        )

        # ----------------------------------------------------
        # ADD FINAL RANK
        # ----------------------------------------------------

        final_results = []

        for rank, result in enumerate(
            ranked_results[:top_k],
            start=1
        ):

            result["rank"] = rank
            final_results.append(result)

        return final_results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MEDICAL HYBRID SEARCH TEST")
    print("=" * 70)

    searcher = HybridSearcher()

    query = "What are common symptoms of asthma?"

    print(f"\nQuery: {query}")
    print("\nTop Hybrid Results:")
    print("-" * 70)

    results = searcher.search(
        query=query,
        top_k=5
    )

    for result in results:

        print(f"\nRank: {result['rank']}")
        print(
            f"Hybrid Score: "
            f"{result['hybrid_score']:.4f}"
        )

        print(
            f"Vector Score: "
            f"{result['vector_score']:.4f}"
        )

        print(
            f"BM25 Score: "
            f"{result['bm25_score']:.4f}"
        )

        print(
            f"Document ID: "
            f"{result['metadata'].get('document_id')}"
        )

        print(
            f"Title: "
            f"{result['metadata'].get('title')}"
        )

        print(
            f"Source: "
            f"{result['metadata'].get('source')}"
        )

        print("\nContent:")
        print(result["content"][:500])

        print("-" * 70)