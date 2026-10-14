from pathlib import Path
import json
from datetime import datetime

from src.generation.rag_pipeline import MedicalRAGPipeline


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
    / "generation_metrics.json"
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
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for keyword-based evaluation.
    """

    return (
        text.lower()
        .replace("-", " ")
        .replace("/", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace(":", " ")
        .replace(";", " ")
    )


# ============================================================
# ANSWER POINT EVALUATION
# ============================================================

def point_is_covered(
    answer,
    expected_point,
):
    """
    Check whether the key concepts of an expected
    answer point appear in the generated answer.
    """

    answer_text = normalize_text(answer)

    keywords = [
        word
        for word in normalize_text(
            expected_point
        ).split()
        if len(word) >= 4
    ]

    if not keywords:
        return False

    matched = sum(
        keyword in answer_text
        for keyword in keywords
    )

    coverage_ratio = (
        matched / len(keywords)
    )

    return coverage_ratio >= 0.5


def calculate_answer_point_coverage(
    answer,
    expected_points,
):
    """
    Calculate semantic-style keyword coverage
    for expected answer points.
    """

    matched_points = []
    missing_points = []

    for point in expected_points:

        if point_is_covered(
            answer,
            point,
        ):
            matched_points.append(point)
        else:
            missing_points.append(point)

    total_points = len(expected_points)

    if total_points == 0:
        coverage = 0.0
    else:
        coverage = (
            len(matched_points)
            / total_points
        )

    return {
        "coverage": coverage,
        "matched_points": matched_points,
        "missing_points": missing_points,
    }


# ============================================================
# CITATION EVALUATION
# ============================================================

def evaluate_citations(
    result,
    expected_documents,
):
    """
    Evaluate citation presence and correctness.
    """

    citations = result.get(
        "citations",
        [],
    )

    cited_documents = []

    for citation in citations:

        document_id = citation.get(
            "document_id"
        )

        if (
            document_id
            and document_id not in cited_documents
        ):
            cited_documents.append(
                document_id
            )

    expected_set = set(
        expected_documents
    )

    cited_set = set(
        cited_documents
    )

    relevant_citations = (
        expected_set.intersection(
            cited_set
        )
    )

    return {
        "citation_count": len(citations),
        "cited_documents": cited_documents,
        "relevant_citations": list(
            relevant_citations
        ),
        "citation_correct": bool(
            relevant_citations
        ),
    }


# ============================================================
# SINGLE QUESTION
# ============================================================

def evaluate_question(
    pipeline,
    query_data,
):
    """
    Evaluate one golden question.
    """

    result = pipeline.process(
        question=query_data["question"],
        top_k=TOP_K,
    )

    answer = result.get(
        "answer",
        "",
    )

    coverage_result = (
        calculate_answer_point_coverage(
            answer,
            query_data[
                "expected_answer_points"
            ],
        )
    )

    citation_result = evaluate_citations(
        result,
        query_data[
            "relevant_documents"
        ],
    )

    return {
        "id": query_data["id"],
        "question": query_data["question"],
        "category": query_data["category"],
        "expected_behavior": query_data[
            "expected_behavior"
        ],
        "expected_answer_points": query_data[
            "expected_answer_points"
        ],
        "matched_answer_points": coverage_result[
            "matched_points"
        ],
        "missing_answer_points": coverage_result[
            "missing_points"
        ],
        "answer_point_coverage": coverage_result[
            "coverage"
        ],
        "answer": answer,
        "citations": result.get(
            "citations",
            []
        ),
        "citation_count": citation_result[
            "citation_count"
        ],
        "cited_documents": citation_result[
            "cited_documents"
        ],
        "relevant_citations": citation_result[
            "relevant_citations"
        ],
        "citation_correct": citation_result[
            "citation_correct"
        ],
        "evidence_sufficient": result.get(
            "evidence_sufficient",
            False,
        ),
        "safety_mode": result.get(
            "safety_mode",
            "",
        ),
    }


# ============================================================
# OVERALL METRICS
# ============================================================

def calculate_overall_metrics(results):
    """
    Calculate overall generation metrics.
    """

    total = len(results)

    if total == 0:
        return {
            "average_answer_point_coverage": 0.0,
            "citation_presence_rate": 0.0,
            "citation_correctness_rate": 0.0,
            "evidence_sufficient_rate": 0.0,
        }

    return {
        "average_answer_point_coverage": sum(
            result["answer_point_coverage"]
            for result in results
        ) / total,

        "citation_presence_rate": sum(
            result["citation_count"] > 0
            for result in results
        ) / total,

        "citation_correctness_rate": sum(
            result["citation_correct"]
            for result in results
        ) / total,

        "evidence_sufficient_rate": sum(
            result["evidence_sufficient"]
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
    Save generation evaluation results.
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
            "top_k": TOP_K,
            "evaluation_method": (
                "keyword_concept_matching"
            ),
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
    print("MEDICAL RAG GENERATION EVALUATION")
    print("=" * 70)

    print("\nLoading golden dataset...")

    golden_dataset = load_golden_dataset()

    print(
        f"Questions loaded : "
        f"{len(golden_dataset)}"
    )

    print("\nInitializing RAG pipeline...")

    pipeline = MedicalRAGPipeline()

    results = []

    for query_data in golden_dataset:

        print(
            f"\nEvaluating "
            f"{query_data['id']}..."
        )

        result = evaluate_question(
            pipeline,
            query_data,
        )

        results.append(result)

        print(
            f"Answer coverage : "
            f"{result['answer_point_coverage'] * 100:.1f}%"
        )

        print(
            f"Citations       : "
            f"{result['citation_count']}"
        )

        print(
            f"Citation correct: "
            f"{result['citation_correct']}"
        )

        print(
            f"Evidence        : "
            f"{result['evidence_sufficient']}"
        )

    overall_metrics = calculate_overall_metrics(
        results
    )

    print("\n" + "=" * 70)
    print("OVERALL GENERATION METRICS")
    print("=" * 70)

    print(
        f"\nAnswer Point Coverage : "
        f"{overall_metrics['average_answer_point_coverage'] * 100:.2f}%"
    )

    print(
        f"Citation Presence    : "
        f"{overall_metrics['citation_presence_rate'] * 100:.2f}%"
    )

    print(
        f"Citation Correctness  : "
        f"{overall_metrics['citation_correctness_rate'] * 100:.2f}%"
    )

    print(
        f"Evidence Sufficient   : "
        f"{overall_metrics['evidence_sufficient_rate'] * 100:.2f}%"
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
    print("GENERATION EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()