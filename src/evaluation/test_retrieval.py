from src.retrieval.vector_search import VectorSearcher
from src.retrieval.keyword_search import BM25Searcher
from src.retrieval.hybrid_search import HybridSearcher


TOP_K = 5


TEST_QUERIES = [
    {
        "query": "What are common symptoms of asthma?",
        "expected_document": "MED-001",
        "expected_terms": [
            "wheezing",
            "cough",
            "shortness of breath",
            "chest tightness",
        ],
    },
    {
        "query": "What are the symptoms of diabetes?",
        "expected_document": "MED-002",
        "expected_terms": [
            "thirst",
            "urination",
            "hunger",
            "fatigue",
        ],
    },
    {
        "query": "What causes GERD?",
        "expected_document": "MED-004",
        "expected_terms": [
            "stomach",
            "acid",
            "esophagus",
            "reflux",
        ],
    },
    {
        "query": "What are migraine symptoms?",
        "expected_document": "MED-005",
        "expected_terms": [
            "headache",
            "nausea",
            "vomiting",
            "light",
        ],
    },
    {
        "query": "What are the symptoms of pneumonia?",
        "expected_document": "MED-008",
        "expected_terms": [
            "cough",
            "fever",
            "breathing",
            "chest",
        ],
    },
    {
        "query": "How can influenza be prevented?",
        "expected_document": "MED-009",
        "expected_terms": [
            "vaccine",
            "vaccination",
            "hand",
            "prevent",
        ],
    },
    {
        "query": "What causes food poisoning?",
        "expected_document": "MED-010",
        "expected_terms": [
            "bacteria",
            "virus",
            "food",
            "contamination",
        ],
    },
    {
        "query": "What medication safety information is available?",
        "expected_document": "MED-012",
        "expected_terms": [
            "safety",
            "drug",
            "medication",
            "warning",
        ],
    },
]


def content_is_relevant(result, expected_terms):
    """
    Check whether a retrieved chunk contains
    at least two expected information terms.
    """

    content = result["content"].lower()

    matched_terms = [
        term
        for term in expected_terms
        if term.lower() in content
    ]

    return len(matched_terms) >= 2


def document_hit(results, expected_document):
    """
    Check whether the expected document appears
    anywhere in the retrieved results.
    """

    documents = [
        result["metadata"].get("document_id")
        for result in results
    ]

    return expected_document in documents


def content_hit(results, expected_terms):
    """
    Check whether at least one retrieved chunk
    contains relevant answer-bearing content.
    """

    return any(
        content_is_relevant(result, expected_terms)
        for result in results
    )


def precision_at_k(results, expected_terms):
    """
    Percentage of retrieved chunks that contain
    relevant answer-bearing content.
    """

    if not results:
        return 0.0

    relevant_count = sum(
        content_is_relevant(result, expected_terms)
        for result in results
    )

    return relevant_count / len(results)


def reciprocal_rank(results, expected_terms):
    """
    Reciprocal rank of the first relevant chunk.
    """

    for rank, result in enumerate(results, start=1):

        if content_is_relevant(
            result,
            expected_terms
        ):
            return 1 / rank

    return 0.0


def run_test(
    searcher,
    test_case,
):
    """
    Run one retrieval test and calculate
    content-level metrics.
    """

    results = searcher.search(
        query=test_case["query"],
        top_k=TOP_K,
    )

    expected_document = test_case["expected_document"]
    expected_terms = test_case["expected_terms"]

    doc_hit = document_hit(
        results,
        expected_document,
    )

    content_hit_result = content_hit(
        results,
        expected_terms,
    )

    precision = precision_at_k(
        results,
        expected_terms,
    )

    mrr = reciprocal_rank(
        results,
        expected_terms,
    )

    retrieved_documents = [
        result["metadata"].get("document_id")
        for result in results
    ]

    return {
        "document_hit": doc_hit,
        "content_hit": content_hit_result,
        "precision": precision,
        "mrr": mrr,
        "retrieved_documents": retrieved_documents,
        "results": results,
    }


def evaluate_system(
    system_name,
    searcher,
):
    """
    Evaluate one retrieval system
    across all test queries.
    """

    print("\n" + "=" * 70)
    print(f"{system_name.upper()} SEARCH")
    print("=" * 70)

    document_hits = 0
    content_hits = 0
    total_precision = 0.0
    total_mrr = 0.0

    for test_case in TEST_QUERIES:

        evaluation = run_test(
            searcher,
            test_case,
        )

        if evaluation["document_hit"]:
            document_hits += 1

        if evaluation["content_hit"]:
            content_hits += 1

        total_precision += evaluation["precision"]
        total_mrr += evaluation["mrr"]

        print("\n" + "-" * 70)
        print(f"Query: {test_case['query']}")
        print(
            f"Expected Document: "
            f"{test_case['expected_document']}"
        )
        print(
            f"Retrieved Documents: "
            f"{evaluation['retrieved_documents']}"
        )

        print(
            f"Document Hit@5 : "
            f"{'PASS' if evaluation['document_hit'] else 'FAIL'}"
        )

        print(
            f"Content Hit@5  : "
            f"{'PASS' if evaluation['content_hit'] else 'FAIL'}"
        )

        print(
            f"Precision@5    : "
            f"{evaluation['precision']:.2f}"
        )

        print(
            f"MRR@5          : "
            f"{evaluation['mrr']:.2f}"
        )

        print("\nTop Retrieved Content:")

        for result in evaluation["results"]:

            content_preview = (
                result["content"]
                .replace("\n", " ")
                .strip()
            )

            print(
                f"\nRank {result['rank']} | "
                f"{result['metadata'].get('document_id')}"
            )

            print(
                content_preview[:250]
                + "..."
            )

    total_queries = len(TEST_QUERIES)

    average_precision = (
        total_precision / total_queries
    )

    average_mrr = (
        total_mrr / total_queries
    )

    print("\n" + "=" * 70)
    print(f"{system_name.upper()} SUMMARY")
    print("=" * 70)

    print(
        f"Document Hit@5 : "
        f"{document_hits}/{total_queries} "
        f"({document_hits / total_queries * 100:.1f}%)"
    )

    print(
        f"Content Hit@5  : "
        f"{content_hits}/{total_queries} "
        f"({content_hits / total_queries * 100:.1f}%)"
    )

    print(
        f"Precision@5    : "
        f"{average_precision:.3f}"
    )

    print(
        f"MRR@5          : "
        f"{average_mrr:.3f}"
    )

    return {
        "document_hit": document_hits / total_queries,
        "content_hit": content_hits / total_queries,
        "precision": average_precision,
        "mrr": average_mrr,
    }


def main():

    print("=" * 70)
    print("MEDICAL RETRIEVAL CONTENT-LEVEL VALIDATION")
    print("=" * 70)

    print("\nInitializing search systems...")

    vector_searcher = VectorSearcher()
    bm25_searcher = BM25Searcher()
    hybrid_searcher = HybridSearcher()

    systems = {
        "Vector": vector_searcher,
        "BM25": bm25_searcher,
        "Hybrid": hybrid_searcher,
    }

    all_results = {}

    for system_name, searcher in systems.items():

        all_results[system_name] = evaluate_system(
            system_name,
            searcher,
        )

    print("\n" + "=" * 70)
    print("FINAL RETRIEVAL COMPARISON")
    print("=" * 70)

    print(
        f"\n{'System':<12}"
        f"{'Doc Hit@5':<14}"
        f"{'Content Hit@5':<17}"
        f"{'Precision@5':<15}"
        f"{'MRR@5':<10}"
    )

    print("-" * 70)

    for system_name, metrics in all_results.items():

        print(
            f"{system_name:<12}"
            f"{metrics['document_hit'] * 100:>6.1f}%       "
            f"{metrics['content_hit'] * 100:>8.1f}%          "
            f"{metrics['precision']:.3f}          "
            f"{metrics['mrr']:.3f}"
        )

    print("\n" + "=" * 70)
    print("CONTENT-LEVEL RETRIEVAL VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()