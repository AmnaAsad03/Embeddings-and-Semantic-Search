from medical_safety import (
    MedicalSafetyChecker,
)


def main():
    checker = MedicalSafetyChecker()

    test_questions = [
        "What are common symptoms of asthma?",
        "What medicine should I take for a headache?",
        "How much should I take?",
        "I am having severe chest pain and difficulty breathing.",
        "I am coughing up blood.",
        "I fainted earlier today.",
        "My symptoms are getting worse.",
        "Can these medicines interact with each other?",
        "Is this medicine safe during pregnancy?",
    ]

    print("=" * 70)
    print("MEDICAL SAFETY CHECK - PHASE 9.3")
    print("=" * 70)

    for question in test_questions:
        result = checker.check_question(question)

        print(f"\nQuestion: {question}")
        print(f"Category: {result.category.value}")
        print(f"Allowed: {result.allowed}")
        print(f"Response mode: {result.response_mode.value}")
        print(f"Severity: {result.severity.value}")
        print(f"Professional care: {result.requires_professional_care}")
        print(f"Message: {result.message}")

    print("\n" + "=" * 70)
    print("ANSWER SAFETY TEST")
    print("=" * 70)

    answers = [
        "Asthma can cause coughing, wheezing, and difficulty breathing.",
        "You should take 2 tablets twice daily.",
    ]

    for answer in answers:
        result = checker.check_answer(answer)

        print(f"\nAnswer: {answer}")
        print(f"Category: {result.category.value}")
        print(f"Allowed: {result.allowed}")
        print(f"Response mode: {result.response_mode.value}")
        print(f"Severity: {result.severity.value}")
        print(f"Message: {result.message}")


if __name__ == "__main__":
    main()