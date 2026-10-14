import json
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking.chunking_strategies import chunk_text


CONFIG_PATH = PROJECT_ROOT / "configs" / "rag_config.yaml"
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "cleaned"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def extract_metadata(text: str) -> dict:
    metadata = {}

    for line in text.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        key = key.strip()
        value = value.strip()

        if key in {
            "document_id",
            "title",
            "source",
            "source_url",
            "topic",
            "document_type",
            "organization",
        }:
            metadata[key] = value.strip('"')

    return metadata


def remove_metadata_block(text: str) -> str:
    lines = text.splitlines()

    content_start = 0

    for i, line in enumerate(lines):
        if line.startswith("# "):
            content_start = i
            break

    return "\n".join(lines[content_start:]).strip()


def create_chunks_for_document(
    file_path: Path,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int
) -> list:

    text = file_path.read_text(encoding="utf-8")

    metadata = extract_metadata(text)

    content = remove_metadata_block(text)

    chunks = chunk_text(
        content,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    results = []

    for index, chunk in enumerate(chunks):

        chunk_id = (
            f"{metadata.get('document_id', file_path.stem)}"
            f"_chunk_{index:04d}"
        )

        results.append(
            {
                "chunk_id": chunk_id,
                "document_id": metadata.get(
                    "document_id",
                    file_path.stem
                ),
                "chunk_index": index,
                "text": chunk,
                "metadata": metadata,
            }
        )

    return results


def main():

    config = load_config()

    chunk_config = config["chunking"]
    output_config = config["output"]

    strategy = chunk_config["strategy"]
    chunk_size = chunk_config["chunk_size"]
    chunk_overlap = chunk_config["chunk_overlap"]

    output_dir = PROJECT_ROOT / output_config["chunks_dir"]
    output_file = PROJECT_ROOT / output_config["chunks_file"]

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    documents = sorted(INPUT_DIR.glob("*.md"))

    if not documents:
        raise FileNotFoundError(
            f"No Markdown documents found in {INPUT_DIR}"
        )

    all_chunks = []

    print("=" * 70)
    print("MEDICAL DOCUMENT CHUNKING")
    print("=" * 70)

    print(f"Documents found : {len(documents)}")
    print(f"Strategy        : {strategy}")
    print(f"Chunk size      : {chunk_size}")
    print(f"Chunk overlap   : {chunk_overlap}")
    print()

    for file_path in documents:

        document_chunks = create_chunks_for_document(
            file_path,
            strategy,
            chunk_size,
            chunk_overlap
        )

        all_chunks.extend(document_chunks)

        print(
            f"{file_path.name:<45} "
            f"{len(document_chunks):>5} chunks"
        )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    sizes = [
        len(chunk["text"])
        for chunk in all_chunks
    ]

    average_size = (
        sum(sizes) / len(sizes)
        if sizes
        else 0
    )

    print()
    print("=" * 70)
    print("CHUNKING COMPLETE")
    print("=" * 70)

    print(f"Total documents : {len(documents)}")
    print(f"Total chunks    : {len(all_chunks)}")
    print(f"Average size    : {average_size:.0f} characters")
    print(f"Output          : {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()