from pathlib import Path
import re
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "cleaned"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "preprocessing_validation_report.json"


REQUIRED_METADATA = [
    "document_id",
    "title",
    "source",
    "source_url",
    "topic",
    "document_type",
    "organization",
]


def extract_metadata(text: str) -> dict:
    metadata = {}

    for field in REQUIRED_METADATA:
        pattern = rf'^{field}:\s*["\']?(.*?)["\']?\s*$'
        match = re.search(
            pattern,
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )

        if match:
            metadata[field] = match.group(1).strip()

    return metadata


def validate_document(file_path: Path) -> dict:
    text = file_path.read_text(encoding="utf-8")
    metadata = extract_metadata(text)

    issues = []

    if not text.strip():
        issues.append("Empty document")

    missing_metadata = [
        field
        for field in REQUIRED_METADATA
        if not metadata.get(field)
    ]

    if missing_metadata:
        issues.append(
            "Missing metadata: " + ", ".join(missing_metadata)
        )

    if re.search(r"\n{3,}", text):
        issues.append("Excessive blank lines")

    html_patterns = [
        "<html",
        "<body",
        "<script",
        "<iframe",
        "cookie policy",
        "accept cookies",
    ]

    html_artifacts = [
        pattern
        for pattern in html_patterns
        if pattern.lower() in text.lower()
    ]

    if html_artifacts:
        issues.append(
            "Possible web artifacts: "
            + ", ".join(html_artifacts)
        )

    heading_count = len(
        re.findall(r"^#{1,6}\s+.+$", text, flags=re.MULTILINE)
    )

    return {
        "filename": file_path.name,
        "document_id": metadata.get("document_id"),
        "title": metadata.get("title"),
        "characters": len(text),
        "words": len(text.split()),
        "headings": heading_count,
        "issues": issues,
        "status": "valid" if not issues else "review",
    }


def main():
    print("=" * 70)
    print("PREPROCESSING VALIDATION")
    print("=" * 70)

    documents = sorted(PROCESSED_DIR.glob("*.md"))

    if not documents:
        print("ERROR: No processed Markdown documents found.")
        print(f"Expected location: {PROCESSED_DIR}")
        return

    results = [
        validate_document(document)
        for document in documents
    ]

    document_ids = [
        result["document_id"]
        for result in results
        if result["document_id"]
    ]

    duplicate_ids = sorted(
        {
            doc_id
            for doc_id in document_ids
            if document_ids.count(doc_id) > 1
        }
    )

    if duplicate_ids:
        for result in results:
            if result["document_id"] in duplicate_ids:
                result["issues"].append(
                    f"Duplicate document_id: {result['document_id']}"
                )
                result["status"] = "review"

    for result in results:
        status = "OK" if result["status"] == "valid" else "REVIEW"

        print(
            f"[{status}] "
            f"{result['filename']} | "
            f"{result['headings']} headings | "
            f"{result['words']:,} words"
        )

        for issue in result["issues"]:
            print(f"      - {issue}")

    valid_documents = sum(
        result["status"] == "valid"
        for result in results
    )

    review_documents = len(results) - valid_documents

    report = {
        "total_documents": len(results),
        "valid_documents": valid_documents,
        "documents_needing_review": review_documents,
        "duplicate_document_ids": duplicate_ids,
        "documents": results,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 70)
    print("PREPROCESSING VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Documents checked: {len(results)}")
    print(f"Valid documents: {valid_documents}")
    print(f"Documents needing review: {review_documents}")
    print(f"Validation report: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()