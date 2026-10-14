from question_validator import QuestionValidator


validator = QuestionValidator()

print("=" * 70)
print("MEDICAL KNOWLEDGE ASSISTANT - INTERACTIVE SAFETY TEST")
print("Type 'exit' to stop")
print("=" * 70)

while True:
    question = input("\nUser: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    result = validator.validate(question)

    print(f"Category : {result.category.value}")
    print(f"Action   : {result.action.value}")
    print(f"Response : {result.message}")