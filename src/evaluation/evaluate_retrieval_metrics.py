from pathlib import Path
import json
from datetime import datetime

from src.retrieval.hybrid_search import HybridSearcher


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLDEN_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "golden_dataset.json"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "experiments"
)

RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "retrieval_metrics.json"
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
    Load the golden evaluation dataset.
    """

    with GOLDEN_DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# RETRIEVAL HELPERS
# ============================================================

def get_retrieved_documents(results):
    """
    Extract unique document IDs from retrieved chunks.
    """

    document_ids = []

    for result in results:

        document_id = result["metadata"].get(
            "document_id"
        )

        if (
            document_id
            and document_id not in document_ids
        ):
            document_ids.append(document_id)

    return document_ids


# ============================================================
# METRICS
# ============================================================

def precision_at_k(
    retrieved_documents,
    relevant_documents,
):
    """
    Calculate Precision@K.

    Precision = relevant retrieved documents / K
    """

    if not retrieved_documents:
        return 0.0

    relevant_count = sum(
        document_id in relevant_documents
        for document_id in retrieved_documents
    )

    return relevant_count / len(retrieved_documents)


def recall_at_k(
    retrieved_documents,
    relevant_documents,
):
    """
    Calculate Recall@K.

    Recall = relevant retrieved documents /
             total relevant documents
    """

    if not relevant_documents:
        return 0.0

    relevant_count = sum(
        document_id in relevant_documents
        for document_id in retrieved_documents
    )

    return relevant_count / len(
        relevant_documents
    )


def f1_at_k(
    precision,
    recall,
):
    """
    Calculate F1@K.
    """

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


def hit_rate_at_k(
    retrieved_documents,
    relevant_documents,
):
    """
    Return 1 if at least one relevant document
    was retrieved.
    """

    if not relevant_documents:
        return 0.0

    return float(
        any(
            document_id in relevant_documents
            for document_id in retrieved_documents
        )
    )


def reciprocal_rank_at_k(
    retrieved_documents,
    relevant_documents,
):
    """
    Calculate Reciprocal Rank.

    Example:
    relevant document at rank 1 -> 1.0
    relevant document at rank 2 -> 0.5
    relevant document at rank 3 -> 0.333
    """

    for rank, document_id in enumerate(
        retrieved_documents,
        start=1,
    ):

        if document_id in relevant_documents:
            return 1.0 / rank

    return 0.0


# ============================================================
# QUESTION EVALUATION
# ============================================================

def evaluate_question(
    retriever,
    query_data,
):
    """
    Evaluate one golden question.
    """

    results = retriever.search(
        query=query_data["question"],
        top_k=TOP_K,
    )

    retrieved_documents = get_retrieved_documents(
        results
    )

    relevant_documents = query_data[
        "relevant_documents"
    ]

    precision = precision_at_k(
        retrieved_documents,
        relevant_documents,
    )

    recall = recall_at_k(
        retrieved_documents,
        relevant_documents,
    )

    f1 = f1_at_k(
        precision,
        recall,
    )

    hit_rate = hit_rate_at_k(
        retrieved_documents,
        relevant_documents,
    )

    reciprocal_rank = reciprocal_rank_at_k(
        retrieved_documents,
        relevant_documents,
    )

    return {
        "id": query_data["id"],
        "question": query_data["question"],
        "category": query_data["category"],
        "relevant_documents": relevant_documents,
        "retrieved_documents": retrieved_documents,
        "precision_at_5": precision,
        "recall_at_5": recall,
        "f1_at_5": f1,
        "hit_rate_at_5": hit_rate,
        "reciprocal_rank_at_5": reciprocal_rank,
    }


# ============================================================
# OVERALL METRICS
# ============================================================

def calculate_overall_metrics(results):
    """
    Calculate macro-average retrieval metrics.
    """

    total = len(results)

    if total == 0:
        return {
            "precision_at_5": 0.0,
            "recall_at_5": 0.0,
            "f1_at_5": 0.0,
            "hit_rate_at_5": 0.0,
            "mrr_at_5": 0.0,
        }

    return {
        "precision_at_5": sum(
            result["precision_at_5"]
            for result in results
        ) / total,

        "recall_at_5": sum(
            result["recall_at_5"]
            for result in results
        ) / total,

        "f1_at_5": sum(
            result["f1_at_5"]
            for result in results
        ) / total,

        "hit_rate_at_5": sum(
            result["hit_rate_at_5"]
            for result in results
        ) / total,

        "mrr_at_5": sum(
            result["reciprocal_rank_at_5"]
            for result in results
        ) / total,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    overall_metrics,
):
    """
    Save detailed and overall retrieval metrics.
    """

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "evaluation": {
            "timestamp": datetime.now().isoformat(),
            "dataset": str(
                GOLDEN_DATASET_PATH
            ),
            "retrieval_method": "hybrid",
            "top_k": TOP_K,
            "evaluation_level": "document",
        },
        "overall_metrics": overall_metrics,
        "per_question": results,
    }

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MEDICAL RAG RETRIEVAL METRICS")
    print("=" * 70)

    print("\nLoading golden dataset...")

    golden_dataset = load_golden_dataset()

    print(
        f"Questions loaded : "
        f"{len(golden_dataset)}"
    )

    print("\nInitializing hybrid retriever...")

    retriever = HybridSearcher()

    results = []

    for query_data in golden_dataset:

        print(
            f"\nEvaluating "
            f"{query_data['id']}..."
        )

        result = evaluate_question(
            retriever,
            query_data,
        )

        results.append(result)

        print(
            f"Precision@5 : "
            f"{result['precision_at_5']:.3f}"
        )

        print(
            f"Recall@5    : "
            f"{result['recall_at_5']:.3f}"
        )

        print(
            f"F1@5        : "
            f"{result['f1_at_5']:.3f}"
        )

        print(
            f"Hit Rate@5  : "
            f"{result['hit_rate_at_5']:.3f}"
        )

        print(
            f"RR@5        : "
            f"{result['reciprocal_rank_at_5']:.3f}"
        )

    overall_metrics = calculate_overall_metrics(
        results
    )

    print("\n" + "=" * 70)
    print("OVERALL RETRIEVAL METRICS")
    print("=" * 70)

    print(
        f"\nPrecision@5 : "
        f"{overall_metrics['precision_at_5']:.4f}"
    )

    print(
        f"Recall@5    : "
        f"{overall_metrics['recall_at_5']:.4f}"
    )

    print(
        f"F1@5        : "
        f"{overall_metrics['f1_at_5']:.4f}"
    )

    print(
        f"Hit Rate@5  : "
        f"{overall_metrics['hit_rate_at_5']:.4f}"
    )

    print(
        f"MRR@5       : "
        f"{overall_metrics['mrr_at_5']:.4f}"
    )

    save_results(
        results,
        overall_metrics,
    )

    print(
        f"\nResults saved to:"
        f"\n{RESULTS_PATH}"
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL METRICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()