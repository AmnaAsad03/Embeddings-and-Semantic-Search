from src.generation.rag_pipeline import MedicalRAGPipeline
from src.safety.medical_safety import ResponseMode


# ============================================================
# FAKE GENERATOR
# ============================================================

class FakeGenerator:

    def __init__(self, answer):
        self.answer = answer

    def generate(self, question, results):

        return {
            "answer": self.answer,
            "citations": [],
            "evidence_sufficient": True,
        }


# ============================================================
# TEST HELPER
# ============================================================

def run_test(name, answer):

    pipeline = MedicalRAGPipeline()

    pipeline.generator = FakeGenerator(answer)

    result = pipeline.process(
        "What are common symptoms of asthma?"
    )

    print("=" * 70)
    print(name)
    print("=" * 70)

    print("\nGenerated answer:")
    print(answer)

    print("\nFinal response:")
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

    return result


# ============================================================
# MAIN TESTS
# ============================================================

def main():

    # ========================================================
    # 1. SAFE RESPONSE
    # ========================================================

    result = run_test(
        "TEST 1 - SAFE RESPONSE",
        "Asthma can cause coughing, wheezing, and difficulty breathing.",
    )

    assert result["safety_allowed"] is True
    assert result["safety_mode"] == ResponseMode.ANSWER.value
    assert result["safety_category"] == "safe"

    print("\n[PASS] Safe response was allowed.")

    # ========================================================
    # 2. ANSWER WITH WARNING
    # ========================================================

    result = run_test(
        "TEST 2 - ANSWER WITH WARNING",
        "This medication may interact with other medicines.",
    )

    assert result["safety_allowed"] is True
    assert result["safety_mode"] == ResponseMode.ANSWER_WITH_WARNING.value
    assert result["safety_category"] == "medication_safety"

    print("\n[PASS] Medication-safety response was allowed with warning.")

    # ========================================================
    # 3. UNSAFE DOSING RESPONSE
    # ========================================================

    result = run_test(
        "TEST 3 - REFUSE UNSAFE DOSING RESPONSE",
        "You should take 2 tablets twice daily.",
    )

    assert result["safety_allowed"] is False
    assert result["safety_mode"] == ResponseMode.REFUSE.value
    assert result["safety_category"] == "unsafe_request"
    assert result["citations"] == []

    print("\n[PASS] Unsafe dosing response was blocked.")

    # ========================================================
    # 4. SELF-TREATMENT RESPONSE
    # ========================================================

    result = run_test(
        "TEST 4 - REFUSE SELF-TREATMENT RESPONSE",
        "There is no need to see a doctor for this condition.",
    )

    assert result["safety_allowed"] is False
    assert result["safety_mode"] == ResponseMode.REFUSE.value
    assert result["safety_category"] == "unsafe_request"
    assert result["citations"] == []

    print("\n[PASS] Unsafe self-treatment response was blocked.")

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("PHASE 9.7 SAFETY RESPONSE TESTS")
    print("=" * 70)

    print("\n[PASS] All Phase 9.7 safety response tests passed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()