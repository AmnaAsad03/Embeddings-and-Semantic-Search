import json
from pathlib import Path


DATASET_PATH = Path("data/evaluation/golden_dataset.json")


def load_dataset():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_golden_dataset_exists():
    assert DATASET_PATH.exists()


def test_golden_dataset_has_30_or_more_questions():
    dataset = load_dataset()

    assert len(dataset) >= 30


def test_question_ids_are_unique():
    dataset = load_dataset()

    ids = [item["id"] for item in dataset]

    assert len(ids) == len(set(ids))


def test_required_fields_exist():
    dataset = load_dataset()

    required_fields = {
        "id",
        "question",
        "category",
        "expected_behavior",
        "relevant_documents",
        "relevant_sections",
        "expected_answer_points",
        "safety_level",
    }

    for item in dataset:
        assert required_fields.issubset(item.keys())


def test_referenced_documents_are_valid():
    dataset = load_dataset()

    valid_documents = {f"MED-{i:03d}" for i in range(1, 13)}

    for item in dataset:
        for document_id in item["relevant_documents"]:
            assert document_id in valid_documents


def test_questions_are_not_empty():
    dataset = load_dataset()

    for item in dataset:
        assert item["question"].strip()


def test_expected_answer_points_are_not_empty():
    dataset = load_dataset()

    for item in dataset:
        assert len(item["expected_answer_points"]) > 0


def test_expected_behaviors_are_valid():
    dataset = load_dataset()

    valid_behaviors = {
        "answer",
        "refuse_insufficient_evidence",
        "refuse_irrelevant",
        "safety_refusal",
        "safety_answer",
        "urgent_escalation",
    }

    for item in dataset:
        assert item["expected_behavior"] in valid_behaviors