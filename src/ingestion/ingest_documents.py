from pathlib import Path
import re
import json


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "cleaned"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "ingestion_report.json"


def clean_markdown(text: str) -> str:
    """
    Clean unnecessary whitespace while preserving
    useful Markdown structure.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing spaces
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # Replace 3+ consecutive blank lines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces inside blank lines
    text = re.sub(r"\n[ \t]+\n", "\n\n", text)

    return text.strip()


def extract_metadata(text: str) -> dict:
    """
    Extract standardized metadata from the beginning
    of the Markdown document.
    """

    metadata = {}

    metadata_fields = [
        "document_id",
        "title",
        "source",
        "source_url",
        "topic",
        "document_type",
        "organization",
    ]

    for field in metadata_fields:
        pattern = rf'^{field}:\s*["\']?(.*?)["\']?\s*$'
        match = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)

        if match:
            metadata[field] = match.group(1).strip()

    return metadata


def validate_document(file_path: Path, text: str, metadata: dict) -> list:
    """
    Validate basic document requirements.
    """

    errors = []

    required_fields = [
        "document_id",
        "title",
        "source",
        "source_url",
        "topic",
        "document_type",
        "organization",
    ]

    for field in required_fields:
        if field not in metadata or not metadata[field]:
            errors.append(f"Missing metadata: {field}")

    if not text.strip():
        errors.append("Document is empty")

    return errors


def process_document(file_path: Path) -> dict:
    """
    Load, clean, extract metadata, and validate one document.
    """

    text = file_path.read_text(encoding="utf-8")

    cleaned_text = clean_markdown(text)

    metadata = extract_metadata(cleaned_text)

    errors = validate_document(
        file_path,
        cleaned_text,
        metadata,
    )

    output_path = PROCESSED_DIR / file_path.name

    output_path.write_text(
        cleaned_text,
        encoding="utf-8",
    )

    return {
        "filename": file_path.name,
        "document_id": metadata.get("document_id"),
        "title": metadata.get("title"),
        "source": metadata.get("source"),
        "characters": len(cleaned_text),
        "words": len(cleaned_text.split()),
        "errors": errors,
        "status": "valid" if not errors else "invalid",
    }


def main():
    """
    Process all Markdown documents in the dataset.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    documents = sorted(
        DOCUMENTS_DIR.glob("*.md")
    )

    if not documents:
        print("ERROR: No Markdown documents found.")
        print(f"Expected location: {DOCUMENTS_DIR}")
        return

    results = []

    print("=" * 70)
    print("MEDICAL DOCUMENT INGESTION")
    print("=" * 70)

    for document in documents:
        result = process_document(document)
        results.append(result)

        status = "OK" if result["status"] == "valid" else "ERROR"

        print(
            f"[{status}] "
            f"{result['filename']} | "
            f"{result['characters']:,} chars | "
            f"{result['words']:,} words"
        )

        if result["errors"]:
            for error in result["errors"]:
                print(f"      - {error}")

    total_documents = len(results)
    valid_documents = sum(
        result["status"] == "valid"
        for result in results
    )

    report = {
        "total_documents": total_documents,
        "valid_documents": valid_documents,
        "invalid_documents": total_documents - valid_documents,
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
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"Documents found: {total_documents}")
    print(f"Valid documents: {valid_documents}")
    print(
        f"Invalid documents: "
        f"{total_documents - valid_documents}"
    )
    print(f"Processed files: {PROCESSED_DIR}")
    print(f"Report: {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()