from pathlib import Path
import json

from src.retrieval.vector_search import VectorSearcher
from src.retrieval.keyword_search import BM25Searcher
from src.retrieval.hybrid_search import HybridSearcher


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLDEN_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "golden_retrieval.json"
)

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
    / "chunks.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5


# ============================================================
# DATA LOADING
# ============================================================

def load_golden_dataset():
    """
    Load the golden retrieval evaluation dataset.
    """

    with open(
        GOLDEN_DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_content_to_chunk_id():
    """
    Build a mapping from chunk text to chunk ID.

    Retrieval metadata currently does not contain chunk_id,
    so we use the authoritative chunks.json file to recover
    the original chunk ID.
    """

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    return {
        chunk["text"]: chunk["chunk_id"]
        for chunk in chunks
    }


# ============================================================
# RETRIEVAL RESULT MAPPING
# ============================================================

def get_retrieved_chunk_ids(
    results,
    content_to_chunk_id,
):
    """
    Extract chunk IDs from retrieval results.

    First checks metadata for chunk_id.

    If chunk_id is not present in metadata,
    maps the retrieved content back to chunks.json.
    """

    chunk_ids = []

    for result in results:

        # Preferred method
        chunk_id = result["metadata"].get(
            "chunk_id"
        )

        # Fallback method
        if not chunk_id:
            chunk_id = content_to_chunk_id.get(
                result["content"]
            )

        if chunk_id:
            chunk_ids.append(chunk_id)

    return chunk_ids


# ============================================================
# METRICS
# ============================================================

def calculate_hit_at_k(
    retrieved_chunk_ids,
    relevant_chunk_ids,
):
    """
    Hit@K = 1 if at least one relevant chunk
    appears in the retrieved results.
    """

    return int(
        any(
            chunk_id in relevant_chunk_ids
            for chunk_id in retrieved_chunk_ids
        )
    )


def calculate_precision_at_k(
    retrieved_chunk_ids,
    relevant_chunk_ids,
):
    """
    Precision@K = relevant retrieved chunks / K.
    """

    if not retrieved_chunk_ids:
        return 0.0

    relevant_count = sum(
        chunk_id in relevant_chunk_ids
        for chunk_id in retrieved_chunk_ids
    )

    return relevant_count / len(retrieved_chunk_ids)


def calculate_reciprocal_rank(
    retrieved_chunk_ids,
    relevant_chunk_ids,
):
    """
    Reciprocal Rank = 1 / rank of the first
    relevant retrieved chunk.
    """

    for rank, chunk_id in enumerate(
        retrieved_chunk_ids,
        start=1,
    ):
        if chunk_id in relevant_chunk_ids:
            return 1 / rank

    return 0.0


# ============================================================
# QUERY EVALUATION
# ============================================================

def evaluate_query(
    searcher,
    query_data,
    content_to_chunk_id,
):
    """
    Evaluate one query against one retrieval system.
    """

    results = searcher.search(
        query=query_data["query"],
        top_k=TOP_K,
    )

    retrieved_chunk_ids = get_retrieved_chunk_ids(
        results,
        content_to_chunk_id,
    )

    relevant_chunk_ids = query_data[
        "relevant_chunk_ids"
    ]

    hit_at_k = calculate_hit_at_k(
        retrieved_chunk_ids,
        relevant_chunk_ids,
    )

    precision_at_k = calculate_precision_at_k(
        retrieved_chunk_ids,
        relevant_chunk_ids,
    )

    reciprocal_rank = calculate_reciprocal_rank(
        retrieved_chunk_ids,
        relevant_chunk_ids,
    )

    return {
        "query_id": query_data["query_id"],
        "query": query_data["query"],
        "expected_document": query_data[
            "expected_document"
        ],
        "relevant_chunk_ids": relevant_chunk_ids,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "hit_at_k": hit_at_k,
        "precision_at_k": precision_at_k,
        "reciprocal_rank": reciprocal_rank,
    }


# ============================================================
# SYSTEM EVALUATION
# ============================================================

def evaluate_system(
    system_name,
    searcher,
    golden_dataset,
    content_to_chunk_id,
):
    """
    Evaluate all golden queries for one retrieval system.
    """

    print("\n" + "=" * 70)
    print(f"{system_name.upper()} RETRIEVAL EVALUATION")
    print("=" * 70)

    results = []

    for query_data in golden_dataset:

        evaluation = evaluate_query(
            searcher,
            query_data,
            content_to_chunk_id,
        )

        results.append(evaluation)

        status = (
            "PASS"
            if evaluation["hit_at_k"]
            else "FAIL"
        )

        print(
            f"\n{evaluation['query_id']}: "
            f"{evaluation['query']}"
        )

        print(
            f"Expected: "
            f"{evaluation['relevant_chunk_ids']}"
        )

        print(
            f"Retrieved: "
            f"{evaluation['retrieved_chunk_ids']}"
        )

        print(
            f"Hit@{TOP_K}: {status}"
        )

        print(
            f"Precision@{TOP_K}: "
            f"{evaluation['precision_at_k']:.3f}"
        )

        print(
            f"RR: "
            f"{evaluation['reciprocal_rank']:.3f}"
        )

    return results


# ============================================================
# SUMMARY METRICS
# ============================================================

def calculate_system_metrics(results):
    """
    Calculate overall retrieval metrics.
    """

    if not results:
        return {
            "hit_at_k": 0.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
        }

    hit_at_k = sum(
        result["hit_at_k"]
        for result in results
    ) / len(results)

    precision_at_k = sum(
        result["precision_at_k"]
        for result in results
    ) / len(results)

    mrr = sum(
        result["reciprocal_rank"]
        for result in results
    ) / len(results)

    return {
        "hit_at_k": hit_at_k,
        "precision_at_k": precision_at_k,
        "mrr": mrr,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GOLDEN RETRIEVAL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load evaluation dataset
    # --------------------------------------------------------

    print("\nLoading golden dataset...")

    golden_dataset = load_golden_dataset()

    print(
        f"Queries loaded : "
        f"{len(golden_dataset)}"
    )

    # --------------------------------------------------------
    # Load authoritative chunk mapping
    # --------------------------------------------------------

    print("\nLoading chunk ID mapping...")

    content_to_chunk_id = (
        load_content_to_chunk_id()
    )

    print(
        f"Chunks mapped  : "
        f"{len(content_to_chunk_id)}"
    )

    # --------------------------------------------------------
    # Initialize retrieval systems
    # --------------------------------------------------------

    print("\nInitializing retrieval systems...")

    vector_searcher = VectorSearcher()

    bm25_searcher = BM25Searcher()

    hybrid_searcher = HybridSearcher()

    # --------------------------------------------------------
    # Evaluate all systems
    # --------------------------------------------------------

    all_results = {}

    all_results["Vector"] = evaluate_system(
        "Vector",
        vector_searcher,
        golden_dataset,
        content_to_chunk_id,
    )

    all_results["BM25"] = evaluate_system(
        "BM25",
        bm25_searcher,
        golden_dataset,
        content_to_chunk_id,
    )

    all_results["Hybrid"] = evaluate_system(
        "Hybrid",
        hybrid_searcher,
        golden_dataset,
        content_to_chunk_id,
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 70)

    print(
        f"\n{'System':<12}"
        f"{'Hit@5':<12}"
        f"{'Precision@5':<16}"
        f"{'MRR@5':<12}"
    )

    print("-" * 52)

    summary = {}

    for system_name, results in all_results.items():

        metrics = calculate_system_metrics(
            results
        )

        summary[system_name] = metrics

        print(
            f"{system_name:<12}"
            f"{metrics['hit_at_k'] * 100:>7.1f}%"
            f"{metrics['precision_at_k']:>15.3f}"
            f"{metrics['mrr']:>12.3f}"
        )

    # --------------------------------------------------------
    # Best system
    # --------------------------------------------------------

    best_system = max(
        summary,
        key=lambda system: (
            summary[system]["hit_at_k"],
            summary[system]["mrr"],
            summary[system]["precision_at_k"],
        ),
    )

    print("\n" + "=" * 70)
    print("BEST RETRIEVAL SYSTEM")
    print("=" * 70)

    print(
        f"\nBest System : {best_system}"
    )

    print(
        f"Hit@5       : "
        f"{summary[best_system]['hit_at_k'] * 100:.1f}%"
    )

    print(
        f"Precision@5 : "
        f"{summary[best_system]['precision_at_k']:.3f}"
    )

    print(
        f"MRR@5       : "
        f"{summary[best_system]['mrr']:.3f}"
    )

    print("\n" + "=" * 70)
    print("GOLDEN RETRIEVAL EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()