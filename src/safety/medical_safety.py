from enum import Enum
from dataclasses import dataclass
import re


class SafetyCategory(str, Enum):
    SAFE = "safe"
    MEDICATION_SAFETY = "medication_safety"
    SELF_MEDICATION = "self_medication"
    EMERGENCY = "emergency"
    UNSAFE_REQUEST = "unsafe_request"


class ResponseMode(str, Enum):
    ANSWER = "answer"
    ANSWER_WITH_WARNING = "answer_with_warning"
    REFUSE = "refuse"
    ESCALATE = "escalate"


class SeverityLevel(str, Enum):
    NONE = "none"
    CAUTION = "caution"
    URGENT = "urgent"
    EMERGENCY = "emergency"


@dataclass
class SafetyResult:
    category: SafetyCategory
    allowed: bool
    message: str
    requires_professional_care: bool = False
    response_mode: ResponseMode = ResponseMode.ANSWER
    severity: SeverityLevel = SeverityLevel.NONE


class MedicalSafetyChecker:
    """
    Medical safety checker for user questions and generated answers.
    """

    UNSAFE_PATTERNS = [
        "how much should i take",
        "how many pills should i take",
        "give me a dosage",
        "what dose should i take",
        "increase my dose",
        "double my dose",
        "take extra pills",
        "can i take more",
        "prescribe me",
        "write me a prescription",
    ]

    DOSAGE_PATTERNS = [
        r"\btake\s+\d+\s+(tablet|tablets|pill|pills|capsule|capsules)",
        r"\b\d+\s+(tablet|tablets|pill|pills|capsule|capsules)\s+(daily|per day|twice daily|three times daily)",
        r"\btake\s+\d+\s*(mg|mcg|g|ml)\b",
        r"\b\d+\s*(mg|mcg|g|ml)\s+(daily|per day|twice daily)",
        r"\b(twice|three times|four times)\s+daily\b",
        r"\bevery\s+\d+\s+hours?\b",
        r"\bevery\s+\d+\s*-\s*\d+\s+hours?\b",
    ]

    PRESCRIPTION_PATTERNS = [
        "you should take",
        "you should start taking",
        "you should stop taking",
        "start taking",
        "stop taking",
        "i recommend taking",
        "i recommend that you take",
        "i prescribe",
        "take this medication",
        "use this medication",
        "you need to take",
        "you must take",
    ]

    SELF_TREATMENT_PATTERNS = [
        "treat it yourself",
        "treat this yourself",
        "self-treat",
        "self treat",
        "you can treat it at home",
        "just treat it at home",
        "no need to see a doctor",
        "you don't need to see a doctor",
        "you do not need to see a doctor",
        "there is no need to see a doctor",
        "avoid seeing a doctor",
        "don't see a doctor",
        "do not see a doctor",
    ]

    DANGEROUS_CERTAINTY_PATTERNS = [
        "will definitely cure",
        "definitely cures",
        "will cure you",
        "definitely cure",
        "guaranteed to cure",
        "there is no risk",
        "completely safe",
        "100% safe",
        "you definitely have",
        "you certainly have",
        "this is definitely",
    ]

    EMERGENCY_PATTERNS = [
        "difficulty breathing",
        "can't breathe",
        "cannot breathe",
        "severe chest pain",
        "chest pain and difficulty breathing",
        "unconscious",
        "not responding",
        "severe bleeding",
        "heavy bleeding",
        "stroke symptoms",
        "seizure",
        "suicide",
        "suicidal",
        "overdose",
        "poisoning",
    ]

    URGENT_PATTERNS = [
        "coughing up blood",
        "vomiting blood",
        "blood in vomit",
        "black stool",
        "fainting",
        "fainted",
        "passed out",
        "confusion",
        "sudden weakness",
        "sudden numbness",
        "severe abdominal pain",
        "severe headache",
        "persistent vomiting",
        "high fever",
        "rapid heartbeat",
    ]

    CAUTION_PATTERNS = [
        "worsening symptoms",
        "getting worse",
        "persistent symptoms",
        "symptoms are not improving",
        "not getting better",
        "new symptoms",
    ]

    SELF_MEDICATION_PATTERNS = [
        "can i take",
        "should i take",
        "can i use",
        "should i use",
        "without seeing a doctor",
        "without a doctor",
        "self medicate",
        "self-medicate",
        "treat myself",
        "what medicine should i take",
        "which medicine should i take",
    ]

    INTERACTION_PATTERNS = [
        "drug interaction",
        "drug interactions",
        "medicine interaction",
        "medicine interactions",
        "medication interaction",
        "medication interactions",
        "interact with",
        "interactions between",
        "can these medicines be taken together",
        "can i take these medicines together",
    ]

    ALLERGY_PATTERNS = [
        "drug allergy",
        "medicine allergy",
        "medication allergy",
        "allergic to",
        "allergy to",
        "allergic reaction",
        "allergic reactions",
    ]

    CONTRAINDICATION_PATTERNS = [
        "contraindication",
        "contraindications",
        "when should i not take",
        "when should i not use",
        "who should not take",
        "who should not use",
        "should not be used",
    ]

    PREGNANCY_AGE_PATTERNS = [
        "pregnant",
        "pregnancy",
        "breastfeeding",
        "breast feeding",
        "for children",
        "for a child",
        "for kids",
        "for elderly",
        "for older adults",
        "for an infant",
        "for infants",
    ]

    def check_question(self, question: str) -> SafetyResult:
        """
        Analyze the user's question before generation.
        """

        question = question.lower().strip()

        if self._contains_any(question, self.EMERGENCY_PATTERNS):
            return SafetyResult(
                category=SafetyCategory.EMERGENCY,
                allowed=False,
                message=(
                    "This may describe a potentially serious or emergency "
                    "situation. Please seek immediate medical attention or "
                    "contact local emergency services. Do not rely on this "
                    "assistant for emergency care."
                ),
                requires_professional_care=True,
                response_mode=ResponseMode.ESCALATE,
                severity=SeverityLevel.EMERGENCY,
            )

        if self._contains_any(question, self.URGENT_PATTERNS):
            return SafetyResult(
                category=SafetyCategory.EMERGENCY,
                allowed=False,
                message=(
                    "These symptoms may require prompt medical assessment. "
                    "Please contact a qualified healthcare professional "
                    "or seek urgent medical care."
                ),
                requires_professional_care=True,
                response_mode=ResponseMode.ESCALATE,
                severity=SeverityLevel.URGENT,
            )

        if self._contains_any(question, self.CAUTION_PATTERNS):
            return SafetyResult(
                category=SafetyCategory.SAFE,
                allowed=True,
                message=(
                    "Persistent or worsening symptoms may require medical "
                    "assessment, especially if they become severe or new "
                    "concerning symptoms appear."
                ),
                requires_professional_care=True,
                response_mode=ResponseMode.ANSWER_WITH_WARNING,
                severity=SeverityLevel.CAUTION,
            )

        if (
            self._contains_any(question, self.UNSAFE_PATTERNS)
            or self._contains_regex(question, self.DOSAGE_PATTERNS)
        ):
            return SafetyResult(
                category=SafetyCategory.UNSAFE_REQUEST,
                allowed=False,
                message=(
                    "I can't provide instructions that could lead to unsafe "
                    "self-medication, medication misuse, or inappropriate "
                    "dosing. Please consult a qualified healthcare "
                    "professional."
                ),
                requires_professional_care=True,
                response_mode=ResponseMode.REFUSE,
                severity=SeverityLevel.CAUTION,
            )

        if self._contains_any(question, self.SELF_MEDICATION_PATTERNS):
            return SafetyResult(
                category=SafetyCategory.SELF_MEDICATION,
                allowed=True,
                message=(
                    "Medication safety depends on factors such as allergies, "
                    "existing conditions, other medicines, age, pregnancy, "
                    "and the specific formulation. Do not start or change "
                    "medication without appropriate professional guidance."
                ),
                requires_professional_care=True,
                response_mode=ResponseMode.ANSWER_WITH_WARNING,
                severity=SeverityLevel.CAUTION,
            )

        if (
            self._contains_any(question, self.INTERACTION_PATTERNS)
            or self._contains_any(question, self.ALLERGY_PATTERNS)
            or self._contains_any(question, self.CONTRAINDICATION_PATTERNS)
            or self._contains_any(question, self.PREGNANCY_AGE_PATTERNS)
        ):
            return SafetyResult(
                category=SafetyCategory.MEDICATION_SAFETY,
                allowed=True,
                message=(
                    "Medication safety can depend on interactions, allergies, "
                    "contraindications, age, pregnancy, breastfeeding, and "
                    "other individual factors. Information should be treated "
                    "as general medical information rather than a personal "
                    "prescription."
                ),
                requires_professional_care=True,
                response_mode=ResponseMode.ANSWER_WITH_WARNING,
                severity=SeverityLevel.CAUTION,
            )

        return SafetyResult(
            category=SafetyCategory.SAFE,
            allowed=True,
            message="No immediate safety concern detected.",
            requires_professional_care=False,
            response_mode=ResponseMode.ANSWER,
            severity=SeverityLevel.NONE,
        )

    def check_answer(self, answer: str) -> SafetyResult:
        """
        Validate generated answer for potentially unsafe medical advice.
        """

        answer = answer.lower().strip()

        # 1. Explicit unsafe dosage instructions
        if self._contains_regex(answer, self.DOSAGE_PATTERNS):
            return SafetyResult(
                category=SafetyCategory.UNSAFE_REQUEST,
                allowed=False,
                message=(
                    "The generated response contained potentially unsafe "
                    "medication or dosing instructions and was blocked."
                ),
                requires_professional_care=True,
                response_mode=ResponseMode.REFUSE,
                severity=SeverityLevel.CAUTION,
            )

        # 2. Prescription-like instructions
        if self._contains_any(answer, self.PRESCRIPTION_PATTERNS):
            return SafetyResult(
                category=SafetyCategory.UNSAFE_REQUEST,
                allowed=False,
                message=(
                    "The generated response contained personalized or "
                    "prescription-like medication instructions and was blocked."
                ),
                requires_professional_care=True,
                response_mode=ResponseMode.REFUSE,
                severity=SeverityLevel.CAUTION,
            )

        # 3. Dangerous self-treatment instructions
        if self._contains_any(answer, self.SELF_TREATMENT_PATTERNS):
            return SafetyResult(
                category=SafetyCategory.UNSAFE_REQUEST,
                allowed=False,
                message=(
                    "The generated response encouraged potentially unsafe "
                    "self-treatment and was blocked."
                ),
                requires_professional_care=True,
                response_mode=ResponseMode.REFUSE,
                severity=SeverityLevel.CAUTION,
            )

        # 4. Overly certain medical claims
        if self._contains_any(answer, self.DANGEROUS_CERTAINTY_PATTERNS):
            return SafetyResult(
                category=SafetyCategory.UNSAFE_REQUEST,
                allowed=False,
                message=(
                    "The generated response contained an overly certain "
                    "medical claim and was blocked."
                ),
                requires_professional_care=True,
                response_mode=ResponseMode.REFUSE,
                severity=SeverityLevel.CAUTION,
            )

        # 5. Existing unsafe request phrases
        if self._contains_any(answer, self.UNSAFE_PATTERNS):
            return SafetyResult(
                category=SafetyCategory.UNSAFE_REQUEST,
                allowed=False,
                message=(
                    "The generated response contained potentially unsafe "
                    "medication instructions and was blocked."
                ),
                requires_professional_care=True,
                response_mode=ResponseMode.REFUSE,
                severity=SeverityLevel.CAUTION,
            )

        # 6. Medication safety information only when an actual
        # medication-safety concept is present.
        if (
            self._contains_any(answer, self.INTERACTION_PATTERNS)
            or self._contains_any(answer, self.ALLERGY_PATTERNS)
            or self._contains_any(answer, self.CONTRAINDICATION_PATTERNS)
            or self._contains_any(answer, self.PREGNANCY_AGE_PATTERNS)
        ):
            return SafetyResult(
                category=SafetyCategory.MEDICATION_SAFETY,
                allowed=True,
                message=(
                    "Medication safety information should be treated as "
                    "general medical information rather than a personal "
                    "prescription. Individual factors and professional "
                    "guidance may be important."
                ),
                requires_professional_care=True,
                response_mode=ResponseMode.ANSWER_WITH_WARNING,
                severity=SeverityLevel.CAUTION,
            )

        # 7. Ordinary medication mentions are safe.
        return SafetyResult(
            category=SafetyCategory.SAFE,
            allowed=True,
            message="Generated answer passed the safety validation.",
            requires_professional_care=False,
            response_mode=ResponseMode.ANSWER,
            severity=SeverityLevel.NONE,
        )

    @staticmethod
    def _contains_any(text: str, patterns: list[str]) -> bool:
        return any(pattern in text for pattern in patterns)

    @staticmethod
    def _contains_regex(text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)