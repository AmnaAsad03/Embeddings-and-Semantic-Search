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
    / "safety_metrics.json"
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
    Load the golden safety evaluation dataset.
    """

    with GOLDEN_DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# EXPECTED BEHAVIOR MAPPING
# ============================================================

def get_actual_behavior(result):
    """
    Convert pipeline output into a normalized
    safety behavior.
    """

    safety_mode = result.get(
        "safety_mode",
        "",
    )

    action = result.get(
        "action",
        "",
    )

    category = result.get(
        "safety_category",
        "",
    )

    # Emergency / escalation
    if safety_mode == "escalate":
        return "urgent_escalation"

    # Safety refusal
    if safety_mode == "refuse":
        return "safety_refusal"

    # Normal answer
    if safety_mode == "answer":
        return "answer"

    # Answer with safety warning
    if safety_mode == "answer_with_warning":
        return "safety_answer"

    # Question validator blocked the request.
    # Preserve this as a separate behavior so we can
    # identify validator-level failures.
    if action != "allow":
        if category == "irrelevant":
            return "refuse_irrelevant"

        if category in {
            "unsupported",
            "insufficient_evidence",
        }:
            return "refuse_insufficient_evidence"

        return "blocked"

    return "unknown"


# ============================================================
# QUESTION EVALUATION
# ============================================================

def evaluate_question(
    pipeline,
    query_data,
):
    """
    Evaluate one question against its expected
    safety behavior.
    """

    result = pipeline.process(
        question=query_data["question"],
        top_k=TOP_K,
    )

    expected_behavior = query_data[
        "expected_behavior"
    ]

    actual_behavior = get_actual_behavior(
        result
    )

    passed = (
        actual_behavior
        == expected_behavior
    )

    return {
        "id": query_data["id"],
        "question": query_data["question"],
        "category": query_data["category"],
        "safety_level": query_data[
            "safety_level"
        ],
        "expected_behavior": expected_behavior,
        "actual_behavior": actual_behavior,
        "passed": passed,
        "action": result.get(
            "action",
            "",
        ),
        "safety_category": result.get(
            "safety_category",
            "",
        ),
        "safety_mode": result.get(
            "safety_mode",
            "",
        ),
        "safety_allowed": result.get(
            "safety_allowed",
            False,
        ),
        "answer": result.get(
            "answer",
            "",
        ),
    }


# ============================================================
# OVERALL METRICS
# ============================================================

def calculate_overall_metrics(results):
    """
    Calculate overall safety metrics.
    """

    total = len(results)

    if total == 0:
        return {
            "total_questions": 0,
            "passed": 0,
            "failed": 0,
            "safety_accuracy": 0.0,
        }

    passed = sum(
        result["passed"]
        for result in results
    )

    failed = total - passed

    return {
        "total_questions": total,
        "passed": passed,
        "failed": failed,
        "safety_accuracy": (
            passed / total
        ),
    }


# ============================================================
# CATEGORY METRICS
# ============================================================

def calculate_category_metrics(results):
    """
    Calculate safety accuracy for each golden
    dataset category.
    """

    categories = {}

    for result in results:

        category = result["category"]

        if category not in categories:
            categories[category] = {
                "total": 0,
                "passed": 0,
            }

        categories[category]["total"] += 1

        if result["passed"]:
            categories[category]["passed"] += 1

    for category, metrics in categories.items():

        metrics["failed"] = (
            metrics["total"]
            - metrics["passed"]
        )

        metrics["accuracy"] = (
            metrics["passed"]
            / metrics["total"]
        )

    return categories


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    overall_metrics,
    category_metrics,
):
    """
    Save safety evaluation results.
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
        },
        "overall_metrics": overall_metrics,
        "category_metrics": category_metrics,
        "failures": [
            result
            for result in results
            if not result["passed"]
        ],
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
    print("MEDICAL RAG SAFETY EVALUATION")
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

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"Expected : "
            f"{result['expected_behavior']}"
        )

        print(
            f"Actual   : "
            f"{result['actual_behavior']}"
        )

        print(
            f"Result   : "
            f"{status}"
        )

    overall_metrics = (
        calculate_overall_metrics(
            results
        )
    )

    category_metrics = (
        calculate_category_metrics(
            results
        )
    )

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("OVERALL SAFETY METRICS")
    print("=" * 70)

    print(
        f"\nTotal Questions : "
        f"{overall_metrics['total_questions']}"
    )

    print(
        f"Passed          : "
        f"{overall_metrics['passed']}"
    )

    print(
        f"Failed          : "
        f"{overall_metrics['failed']}"
    )

    print(
        f"Safety Accuracy : "
        f"{overall_metrics['safety_accuracy'] * 100:.2f}%"
    )

    # ========================================================
    # CATEGORY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("CATEGORY SAFETY RESULTS")
    print("=" * 70)

    for category, metrics in category_metrics.items():

        print(
            f"\n{category}: "
            f"{metrics['passed']}/"
            f"{metrics['total']} "
            f"({metrics['accuracy'] * 100:.1f}%)"
        )

    # ========================================================
    # FAILURES
    # ========================================================

    failures = [
        result
        for result in results
        if not result["passed"]
    ]

    if failures:

        print("\n" + "=" * 70)
        print("SAFETY FAILURES")
        print("=" * 70)

        for result in failures:

            print(
                f"\n{result['id']}"
            )

            print(
                f"Question : "
                f"{result['question']}"
            )

            print(
                f"Expected : "
                f"{result['expected_behavior']}"
            )

            print(
                f"Actual   : "
                f"{result['actual_behavior']}"
            )

            print(
                f"Action   : "
                f"{result['action']}"
            )

            print(
                f"Safety   : "
                f"{result['safety_mode']}"
            )

    save_results(
        results,
        overall_metrics,
        category_metrics,
    )

    print(
        f"\nResults saved to:"
        f"\n{RESULTS_PATH}"
    )

    print("\n" + "=" * 70)
    print("SAFETY EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()