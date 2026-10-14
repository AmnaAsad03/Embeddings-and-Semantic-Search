from src.safety.question_validator import (
    QuestionValidator,
    SafetyAction,
    QuestionCategory,
)

from src.safety.medical_safety import (
    MedicalSafetyChecker,
    ResponseMode,
    SafetyCategory,
    SeverityLevel,
)

from src.retrieval.hybrid_search import HybridSearcher

from src.generation.generator import (
    MedicalAnswerGenerator,
)

from src.generation.citations import (
    format_citations,
)


class MedicalRAGPipeline:

    def __init__(self):

        print("Initializing Medical RAG pipeline...")

        self.validator = QuestionValidator()
        self.safety_checker = MedicalSafetyChecker()
        self.retriever = HybridSearcher()
        self.generator = MedicalAnswerGenerator()

        print("Medical RAG pipeline initialized.")

    def process(self, question, top_k=5):

        # --------------------------------------------------
        # 1. Validate the user's question
        # --------------------------------------------------

        validation = self.validator.validate(question)

        # --------------------------------------------------
        # 2. Handle emergency escalation
        # --------------------------------------------------

        if validation.action == SafetyAction.ESCALATE:

            return {
                "question": question,
                "answer": validation.message,
                "category": validation.category.value,
                "action": validation.action.value,
                "citations": [],
                "evidence_sufficient": False,
                "safety_category": SafetyCategory.EMERGENCY.value,
                "safety_allowed": False,
                "safety_mode": ResponseMode.ESCALATE.value,
                "safety_severity": SeverityLevel.EMERGENCY.value,
            }

        # --------------------------------------------------
        # 3. Handle refusal
        # --------------------------------------------------

        if validation.action == SafetyAction.REFUSE:

            if validation.category == QuestionCategory.DANGEROUS_SELF_MEDICATION:
                safety_category = SafetyCategory.SELF_MEDICATION.value
            elif validation.category == QuestionCategory.INAPPROPRIATE:
                safety_category = SafetyCategory.UNSAFE_REQUEST.value
            elif validation.category == QuestionCategory.EMERGENCY:
                safety_category = SafetyCategory.EMERGENCY.value
            else:
                safety_category = validation.category.value

            return {
                "question": question,
                "answer": validation.message,
                "category": validation.category.value,
                "action": validation.action.value,
                "citations": [],
                "evidence_sufficient": False,
                "safety_category": safety_category,
                "safety_allowed": False,
                "safety_mode": ResponseMode.REFUSE.value,
                "safety_severity": SeverityLevel.CAUTION.value,
            }

        # --------------------------------------------------
        # 4. Retrieve relevant evidence
        # --------------------------------------------------

        results = self.retriever.search(
            query=question,
            top_k=top_k,
        )

        # --------------------------------------------------
        # 5. Generate answer from retrieved evidence
        # --------------------------------------------------

        generation_result = self.generator.generate(
            question=question,
            results=results,
        )

        answer = generation_result["answer"]

        # --------------------------------------------------
        # 6. Safety-check generated answer
        # --------------------------------------------------

        safety_result = self.safety_checker.check_answer(
            answer
        )

        # --------------------------------------------------
        # 7. Refuse unsafe generated answer
        # --------------------------------------------------

        if safety_result.response_mode == ResponseMode.REFUSE:

            return {
                "question": question,
                "answer": safety_result.message,
                "category": validation.category.value,
                "action": validation.action.value,
                "citations": [],
                "evidence_sufficient": False,
                "safety_category": safety_result.category.value,
                "safety_allowed": False,
                "safety_mode": safety_result.response_mode.value,
                "safety_severity": safety_result.severity.value,
            }

        # --------------------------------------------------
        # 8. Escalate serious generated-answer situations
        # --------------------------------------------------

        if safety_result.response_mode == ResponseMode.ESCALATE:

            return {
                "question": question,
                "answer": safety_result.message,
                "category": validation.category.value,
                "action": validation.action.value,
                "citations": [],
                "evidence_sufficient": False,
                "safety_category": safety_result.category.value,
                "safety_allowed": False,
                "safety_mode": safety_result.response_mode.value,
                "safety_severity": safety_result.severity.value,
            }

        # --------------------------------------------------
        # 9. Prepare citations
        # --------------------------------------------------

        citation_text = ""

        if generation_result["evidence_sufficient"]:

            citation_text = format_citations(
                generation_result["citations"]
            )

        # --------------------------------------------------
        # 10. Add safety warning when actually required
        # --------------------------------------------------

        if safety_result.response_mode == ResponseMode.ANSWER_WITH_WARNING:

            answer = (
                f"{safety_result.message}\n\n"
                f"{answer}"
            )

        # --------------------------------------------------
        # 11. Add citations
        # --------------------------------------------------

        if generation_result["evidence_sufficient"]:

            answer = (
                f"{answer}\n\n"
                f"{citation_text}"
            )

        # --------------------------------------------------
        # 12. Return final result
        # --------------------------------------------------

        return {
            "question": question,
            "answer": answer,
            "category": validation.category.value,
            "action": validation.action.value,
            "citations": generation_result["citations"],
            "evidence_sufficient": generation_result[
                "evidence_sufficient"
            ],
            "safety_category": safety_result.category.value,
            "safety_allowed": safety_result.allowed,
            "safety_mode": safety_result.response_mode.value,
            "safety_severity": safety_result.severity.value,
        }