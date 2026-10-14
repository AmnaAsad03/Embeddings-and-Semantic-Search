from pathlib import Path
import json
import re
import math
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
    / "chunks.json"
)

DEFAULT_TOP_K = 5


# ============================================================
# BM25 SEARCHER
# ============================================================

class BM25Searcher:

    def __init__(self, chunks_path=CHUNKS_PATH):

        print("Loading medical chunks...")

        with open(chunks_path, "r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        self.documents = [
            chunk["text"]
            for chunk in self.chunks
        ]

        self.tokenized_documents = [
            self._tokenize(document)
            for document in self.documents
        ]

        self.doc_count = len(self.documents)

        self.doc_lengths = [
            len(tokens)
            for tokens in self.tokenized_documents
        ]

        self.avg_doc_length = (
            sum(self.doc_lengths) / self.doc_count
        )

        self.k1 = 1.5
        self.b = 0.75

        self.document_frequency = (
            self._build_document_frequency()
        )

        print("BM25 search initialized.")
        print(f"Chunks     : {self.doc_count}")
        print(f"Vocabulary : {len(self.document_frequency)}")

    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    def _tokenize(self, text):

        return re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )

    # --------------------------------------------------------
    # DOCUMENT FREQUENCY
    # --------------------------------------------------------

    def _build_document_frequency(self):

        frequency = Counter()

        for tokens in self.tokenized_documents:

            for token in set(tokens):
                frequency[token] += 1

        return frequency

    # --------------------------------------------------------
    # BM25 SEARCH
    # --------------------------------------------------------

    def search(self, query, top_k=DEFAULT_TOP_K):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_tokens = self._tokenize(query)

        scores = []

        for index, document_tokens in enumerate(
            self.tokenized_documents
        ):

            term_frequency = Counter(document_tokens)

            score = 0.0

            for term in query_tokens:

                if term not in term_frequency:
                    continue

                tf = term_frequency[term]
                df = self.document_frequency[term]

                idf = math.log(
                    1
                    + (
                        self.doc_count - df + 0.5
                    )
                    / (
                        df + 0.5
                    )
                )

                document_length = self.doc_lengths[index]

                denominator = (
                    tf
                    + self.k1
                    * (
                        1
                        - self.b
                        + self.b
                        * document_length
                        / self.avg_doc_length
                    )
                )

                score += (
                    idf
                    * (
                        tf * (self.k1 + 1)
                        / denominator
                    )
                )

            scores.append(score)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for rank, index in enumerate(
            ranked_indices[:top_k],
            start=1
        ):

            chunk = self.chunks[index]

            results.append(
                {
                    "rank": rank,
                    "content": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": scores[index],
                }
            )

        return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MEDICAL KEYWORD / BM25 SEARCH TEST")
    print("=" * 70)

    searcher = BM25Searcher()

    query = "What are common symptoms of asthma?"

    print(f"\nQuery: {query}")
    print("\nTop Results:")
    print("-" * 70)

    results = searcher.search(
        query=query,
        top_k=5
    )

    for result in results:

        print(f"\nRank: {result['rank']}")
        print(f"BM25 Score: {result['score']:.4f}")

        print(
            f"Document ID: "
            f"{result['metadata'].get('document_id')}"
        )

        print(
            f"Title: "
            f"{result['metadata'].get('title')}"
        )

        print(
            f"Source: "
            f"{result['metadata'].get('source')}"
        )

        print("\nContent:")
        print(result["content"][:500])

        print("-" * 70)