from src.generation.rag_pipeline import MedicalRAGPipeline


class FakeGenerator:

    def generate(self, question, results):

        return {
            "answer": "You should take 2 tablets twice daily.",
            "citations": [],
            "evidence_sufficient": True,
        }


def main():

    pipeline = MedicalRAGPipeline()

    # Replace the real generator with a controlled unsafe generator.
    pipeline.generator = FakeGenerator()

    result = pipeline.process(
        "What are common symptoms of asthma?"
    )

    print("=" * 70)
    print("PIPELINE SAFETY INTEGRATION TEST")
    print("=" * 70)

    print("\nQuestion:")
    print(result["question"])

    print("\nFinal answer:")
    print(result["answer"])

    print("\nSafety category:")
    print(result["safety_category"])

    print("Safety allowed:")
    print(result["safety_allowed"])

    print("Safety mode:")
    print(result["safety_mode"])

    print("Safety severity:")
    print(result["safety_severity"])

    print("\nCitations:")
    print(result["citations"])

    # ========================================================
    # ASSERTIONS
    # ========================================================

    assert result["safety_allowed"] is False
    assert result["safety_mode"] == "refuse"
    assert result["safety_category"] == "unsafe_request"
    assert result["citations"] == []

    print("\n[PASS] Unsafe generated answer was blocked.")


if __name__ == "__main__":
    main()