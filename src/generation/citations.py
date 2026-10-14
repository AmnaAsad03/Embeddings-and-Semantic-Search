def format_citations(citations):
    """
    Format unique citation metadata into a readable Sources section.
    """

    if not citations:
        return "Sources:\nNo sources available."

    lines = ["Sources:"]
    seen = set()
    citation_number = 1

    for citation in citations:
        document_id = citation.get("document_id")
        title = citation.get("title", "Unknown document")
        source = citation.get("source", "Unknown source")
        source_url = citation.get("source_url")

        # Use document ID when available to identify duplicates.
        unique_key = document_id or source_url or f"{title}|{source}"

        if unique_key in seen:
            continue

        seen.add(unique_key)

        line = f"[{citation_number}] {title} — {source}"

        if source_url:
            line += f"\n    {source_url}"

        lines.append(line)
        citation_number += 1

    return "\n".join(lines)