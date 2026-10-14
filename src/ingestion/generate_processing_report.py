from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INGESTION_REPORT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ingestion_report.json"
)

VALIDATION_REPORT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing_validation_report.json"
)

OUTPUT_REPORT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "processing_report.json"
)


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def main():
    print("=" * 70)
    print("PROCESSING REPORT")
    print("=" * 70)

    ingestion = load_json(INGESTION_REPORT)
    validation = load_json(VALIDATION_REPORT)

    documents = validation["documents"]

    total_words = sum(
        doc["words"]
        for doc in documents
    )

    total_characters = sum(
        doc["characters"]
        for doc in documents
    )

    total_headings = sum(
        doc["headings"]
        for doc in documents
    )

    report = {
        "dataset": {
            "total_documents": validation["total_documents"],
            "valid_documents": validation["valid_documents"],
            "documents_needing_review": validation[
                "documents_needing_review"
            ],
            "duplicate_document_ids": validation[
                "duplicate_document_ids"
            ],
        },
        "content_statistics": {
            "total_words": total_words,
            "total_characters": total_characters,
            "total_headings": total_headings,
            "average_words_per_document": round(
                total_words / len(documents), 2
            ),
            "average_characters_per_document": round(
                total_characters / len(documents), 2
            ),
        },
        "processing": {
            "cleaned_documents": ingestion.get(
                "valid_documents", 0
            ),
            "output_directory": str(
                PROJECT_ROOT
                / "data"
                / "processed"
                / "cleaned"
            ),
        },
        "documents": documents,
    }

    OUTPUT_REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Total documents: {len(documents)}")
    print(f"Total words: {total_words:,}")
    print(f"Total characters: {total_characters:,}")
    print(f"Total headings: {total_headings}")
    print(
        f"Average words/document: "
        f"{total_words / len(documents):,.2f}"
    )
    print(
        f"Average characters/document: "
        f"{total_characters / len(documents):,.2f}"
    )
    print(f"Report: {OUTPUT_REPORT}")

    print("=" * 70)
    print("PROCESSING REPORT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()