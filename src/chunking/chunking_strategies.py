import re
from typing import Dict, List


def fixed_chunking(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> List[str]:
    """Split text into fixed-size overlapping chunks."""

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def recursive_chunking(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> List[str]:
    """Split text using progressively smaller separators."""

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(value: str, separator_index: int = 0) -> List[str]:
        value = value.strip()

        if len(value) <= chunk_size:
            return [value] if value else []

        if separator_index >= len(separators):
            return fixed_chunking(
                value,
                chunk_size,
                chunk_overlap
            )

        separator = separators[separator_index]

        if separator:
            parts = value.split(separator)
        else:
            parts = list(value)

        chunks = []
        current = ""

        for part in parts:
            candidate = (
                part
                if not current
                else current + separator + part
            )

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())

                if len(part) > chunk_size:
                    chunks.extend(
                        split_text(part, separator_index + 1)
                    )
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks

    base_chunks = split_text(text)

    final_chunks = []

    for i, chunk in enumerate(base_chunks):
        if i == 0:
            final_chunks.append(chunk)
            continue

        previous = base_chunks[i - 1]

        overlap_text = previous[-chunk_overlap:]

        combined = (
            overlap_text + " " + chunk
        ).strip()

        if len(combined) > chunk_size:
            combined = combined[-chunk_size:]

        final_chunks.append(combined)

    return final_chunks


def structure_chunking(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> List[str]:
    """Split Markdown while preserving heading/content structure."""

    sections = re.split(
        r"(?=^#{1,6}\s+.+$)",
        text,
        flags=re.MULTILINE
    )

    chunks = []

    for section in sections:
        section = section.strip()

        if not section:
            continue

        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            chunks.extend(
                recursive_chunking(
                    section,
                    chunk_size,
                    chunk_overlap
                )
            )

    return chunks


def semantic_chunking(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> List[str]:
    """
    Placeholder for embedding-based semantic chunking.

    Semantic chunking will be implemented after the
    embedding pipeline is available.
    """

    raise NotImplementedError(
        "Semantic chunking will be implemented "
        "after the embedding pipeline."
    )


def chunk_text(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> List[str]:

    strategies = {
        "fixed": fixed_chunking,
        "recursive": recursive_chunking,
        "structure": structure_chunking,
        "semantic": semantic_chunking,
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}. "
            f"Available: {list(strategies.keys())}"
        )

    return strategies[strategy](
        text,
        chunk_size,
        chunk_overlap
    )