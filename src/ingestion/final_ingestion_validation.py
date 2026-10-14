from pathlib import Path
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = PROJECT_ROOT / "data" / "documents"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "cleaned"

INGESTION_REPORT = (
    PROJECT_ROOT / "data" / "processed" / "ingestion_report.json"
)

VALIDATION_REPORT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing_validation_report.json"
)

PROCESSING_REPORT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "processing_report.json"
)

FINAL_REPORT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_ingestion_validation.json"
)


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
        pattern = rf"^{field}:\s*[\"']?(.*?)[\"']?\s*$"

        match = re.search(
            pattern,
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )

        if match:
            metadata[field] = match.group(1).strip()

    return metadata


def main():
    print("=" * 70)
    print("FINAL INGESTION VALIDATION")
    print("=" * 70)

    source_documents = sorted(
        SOURCE_DIR.glob("*.md")
    )

    processed_documents = sorted(
        PROCESSED_DIR.glob("*.md")
    )

    errors = []

    # ---------------------------------------------------------
    # 1. Document count validation
    # ---------------------------------------------------------

    if len(source_documents) != len(processed_documents):
        errors.append(
            f"Document count mismatch: "
            f"{len(source_documents)} source vs "
            f"{len(processed_documents)} processed"
        )

    # ---------------------------------------------------------
    # 2. Processed document validation
    # ---------------------------------------------------------

    document_ids = []

    for file_path in processed_documents:
        text = file_path.read_text(
            encoding="utf-8"
        )

        if not text.strip():
            errors.append(
                f"Empty processed document: {file_path.name}"
            )
            continue

        metadata = extract_metadata(text)

        missing_metadata = [
            field
            for field in REQUIRED_METADATA
            if not metadata.get(field)
        ]

        if missing_metadata:
            errors.append(
                f"{file_path.name}: missing metadata: "
                + ", ".join(missing_metadata)
            )

        document_id = metadata.get("document_id")

        if document_id:
            document_ids.append(document_id)

    # ---------------------------------------------------------
    # 3. Duplicate ID validation
    # ---------------------------------------------------------

    duplicates = sorted(
        {
            doc_id
            for doc_id in document_ids
            if document_ids.count(doc_id) > 1
        }
    )

    if duplicates:
        errors.append(
            "Duplicate document IDs: "
            + ", ".join(duplicates)
        )

    # ---------------------------------------------------------
    # 4. Required report validation
    # ---------------------------------------------------------

    required_reports = [
        INGESTION_REPORT,
        VALIDATION_REPORT,
        PROCESSING_REPORT,
    ]

    missing_reports = [
        str(report)
        for report in required_reports
        if not report.exists()
    ]

    if missing_reports:
        errors.append(
            "Missing report files: "
            + ", ".join(missing_reports)
        )

    # ---------------------------------------------------------
    # 5. Report consistency validation
    # ---------------------------------------------------------

    if VALIDATION_REPORT.exists():
        validation = json.loads(
            VALIDATION_REPORT.read_text(
                encoding="utf-8"
            )
        )

        if validation.get("total_documents") != len(
            processed_documents
        ):
            errors.append(
                "Validation report document count "
                "does not match processed documents"
            )

        if validation.get("valid_documents") != len(
            processed_documents
        ):
            errors.append(
                "Not all processed documents are marked valid"
            )

        if validation.get("documents_needing_review", 0) != 0:
            errors.append(
                "Some documents still require review"
            )

    # ---------------------------------------------------------
    # 6. Final status
    # ---------------------------------------------------------

    status = "PASS" if not errors else "FAIL"

    report = {
        "status": status,
        "source_documents": len(source_documents),
        "processed_documents": len(processed_documents),
        "unique_document_ids": len(set(document_ids)),
        "errors": errors,
    }

    FINAL_REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 7. Terminal output
    # ---------------------------------------------------------

    print(
        f"Source documents: {len(source_documents)}"
    )

    print(
        f"Processed documents: {len(processed_documents)}"
    )

    print(
        f"Unique document IDs: {len(set(document_ids))}"
    )

    print(
        f"Required reports: "
        f"{'OK' if not missing_reports else 'MISSING'}"
    )

    if errors:
        print("\nVALIDATION ERRORS:")

        for error in errors:
            print(f"[ERROR] {error}")
    else:
        print("\n[PASS] Document counts match")
        print("[PASS] All processed documents are non-empty")
        print("[PASS] All required metadata is present")
        print("[PASS] Document IDs are unique")
        print("[PASS] Required reports exist")
        print("[PASS] Processing reports are consistent")

    print("=" * 70)
    print(f"FINAL VALIDATION: {status}")
    print("=" * 70)
    print(f"Report: {FINAL_REPORT}")
    print("=" * 70)


if __name__ == "__main__":
    main()