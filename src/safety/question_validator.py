import re
from dataclasses import dataclass
from enum import Enum


class QuestionCategory(str, Enum):
    RELEVANT_MEDICAL = "relevant_medical"
    UNRELATED = "unrelated"
    UNSUPPORTED_MEDICAL = "unsupported_medical"
    INAPPROPRIATE = "inappropriate"
    DANGEROUS_SELF_MEDICATION = "dangerous_self_medication"
    EMERGENCY = "emergency"


class SafetyAction(str, Enum):
    ALLOW = "allow"
    REFUSE = "refuse"
    ESCALATE = "escalate"


@dataclass
class ValidationResult:
    category: QuestionCategory
    action: SafetyAction
    message: str


class QuestionValidator:
    """
    Validates user questions before they enter the RAG pipeline.
    """

    MEDICAL_KEYWORDS = {
        "disease",
        "symptom",
        "symptoms",
        "treatment",
        "diagnosis",
        "medicine",
        "medication",
        "drug",
        "health",
        "illness",
        "condition",
        "epilepsy",
        "doctor",
        "hospital",
        "pain",
        "fever",
        "cough",
        "breathing",
        "headache",
        "allergy",
        "allergic",
        "pregnancy",
        "infection",
        "signs",
        "complication",
        "complications",
        "prevent",
        "prevention",
        "asthma",
        "diabetes",
        "migraine",
        "anemia",
        "pneumonia",
        "influenza",
        "flu",
        "gerd",
        "depression",
        "cardiovascular",
        "heart",
        "food",
        "poisoning",
        "respiratory",
    }

    SUPPORTED_TOPICS = {
        "asthma",
        "diabetes",
        "pulmonary hypertension",
        "gerd",
        "gastroesophageal reflux",
        "migraine",
        "anemia",
        "cardiovascular disease",
        "cardiovascular diseases",
        "heart disease",
        "pneumonia",
        "influenza",
        "flu",
        "food poisoning",
        "depression",
        "drug safety",
        "medication safety",
    }

    MEDICATION_SAFETY_PATTERNS = [
        "medication allergy",
        "medicine allergy",
        "drug allergy",
        "allergic to",
        "allergy to",
        "allergic reaction",
        "allergic reactions",
        "drug interaction",
        "drug interactions",
        "medicine interaction",
        "medicine interactions",
        "medication interaction",
        "medication interactions",
        "interact with",
        "interactions between",
        "contraindication",
        "contraindications",
        "when should i not take",
        "when should i not use",
        "who should not take",
        "who should not use",
        "should not be used",
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

    CARDIO_RESPIRATORY_PATTERNS = [
        "cardiovascular and respiratory",
        "cardiovascular or respiratory",
        "cardiovascular system",
        "respiratory system",
        "cardiovascular systems",
        "respiratory systems",
    ]

    EMERGENCY_PATTERNS = [
        r"\bcan't breathe\b",
        r"\bcannot breathe\b",
        r"\bnot breathing\b",
        r"\bsevere difficulty breathing\b",
        r"\bsevere chest pain\b",
        r"\bchest pain\b.*\b(faint|collapse|breath)\b",
        r"\bunconscious\b",
        r"\bface is drooping\b",
        r"\bpassed out\b",
        r"\bseizure\b",
        r"\bsevere bleeding\b",
        r"\bbleeding heavily\b",
        r"\bstroke symptoms?\b",
        r"\bface drooping\b",
        r"\bsudden weakness\b",
        r"\bhaving a stroke\b",
        r"\bthink.*\b(stroke)\b",
        r"\bsuicid(al|e)\b",
        r"\boverdose\b",
        r"\boverdosed\b",
        r"\boverdosing\b",
        r"\btook too much\b.*\b(medicine|medication|drug)\b",
        r"\bpoisoned\b",
        r"\blife-threatening medical emergency\b",
    ]

    SELF_MEDICATION_PATTERNS = [
        r"\bshould i take\b",
        r"\bcan i take\b",
        r"\bwhat medicine should i take\b",
        r"\bwhat medication should i take\b",
        r"\bwhich medicine should i take\b",
        r"\bwhich drug should i take\b",
        r"\bhow much should i take\b",
        r"\bwhat dosage should i take\b",
        r"\bwhat dose should i take\b",
        r"\bincrease my dose\b",
        r"\bincrease my medication dose\b",
        r"\bincrease my medicine dose\b",
        r"\bdecrease my dose\b",
        r"\bdecrease my medication dose\b",
        r"\bdecrease my medicine dose\b",
        r"\bdouble my dose\b",
        r"\bchange my dose\b",
        r"\bchange my medication dose\b",
        r"\bchange my medicine dose\b",
        r"\bstop taking\b.*\bmedicine\b",
        r"\bstop taking\b.*\bmedication\b",
        r"\bstart taking\b.*\bmedicine\b",
        r"\bstart taking\b.*\bmedication\b",
    ]

    INAPPROPRIATE_PATTERNS = [
        r"\bhow (to|can i|can you|do i)\b.*\bmake\b.*\b(drug|poison)\b",
        r"\bhow (to|can i|can you|do i)\b.*\bproduce\b.*\b(drug|poison)\b",
        r"\bhow (to|can i|can you|do i)\b.*\bcreate\b.*\b(drug|poison)\b",
        r"\bhow to abuse\b.*\bdrug\b",
        r"\bhow to overdose\b",
    ]

    def validate(self, question: str) -> ValidationResult:
        question = question.strip()

        if not question:
            return ValidationResult(
                category=QuestionCategory.UNRELATED,
                action=SafetyAction.REFUSE,
                message="Please provide a medical question.",
            )

        normalized = re.sub(r"\s+", " ", question.lower())

        # 1. Harmful or inappropriate requests
        if self._matches(normalized, self.INAPPROPRIATE_PATTERNS):
            return ValidationResult(
                category=QuestionCategory.INAPPROPRIATE,
                action=SafetyAction.REFUSE,
                message=(
                    "I can't provide instructions that could facilitate "
                    "harmful or inappropriate drug use."
                ),
            )

        # 2. Emergency situations
        if self._matches(normalized, self.EMERGENCY_PATTERNS):
            return ValidationResult(
                category=QuestionCategory.EMERGENCY,
                action=SafetyAction.ESCALATE,
                message=(
                    "This may describe a serious or emergency situation. "
                    "Please seek appropriate urgent medical care rather "
                    "than relying on this knowledge assistant."
                ),
            )

        # 3. Dangerous self-medication
        if self._matches(normalized, self.SELF_MEDICATION_PATTERNS):
            return ValidationResult(
                category=QuestionCategory.DANGEROUS_SELF_MEDICATION,
                action=SafetyAction.REFUSE,
                message=(
                    "I can't provide personalized instructions for starting, "
                    "stopping, or changing medication. Medication safety can "
                    "depend on factors such as allergies, medical conditions, "
                    "interactions, age, pregnancy, and dosage requirements."
                ),
            )

        # 4. Medication safety questions
        if self._contains_any(normalized, self.MEDICATION_SAFETY_PATTERNS):
            return ValidationResult(
                category=QuestionCategory.RELEVANT_MEDICAL,
                action=SafetyAction.ALLOW,
                message="Question accepted for medical safety information.",
            )

        # 5. Supported cardiovascular/respiratory questions
        if self._contains_any(normalized, self.CARDIO_RESPIRATORY_PATTERNS):
            return ValidationResult(
                category=QuestionCategory.RELEVANT_MEDICAL,
                action=SafetyAction.ALLOW,
                message="Question accepted for medical knowledge retrieval.",
            )

        # 6. Medical question
        if self._contains_medical_keyword(normalized):

            if self._contains_supported_topic(normalized):
                return ValidationResult(
                    category=QuestionCategory.RELEVANT_MEDICAL,
                    action=SafetyAction.ALLOW,
                    message="Question accepted for medical knowledge retrieval.",
                )

            return ValidationResult(
                category=QuestionCategory.UNSUPPORTED_MEDICAL,
                action=SafetyAction.REFUSE,
                message=(
                    "This is a medical question, but the requested topic is "
                    "not currently covered by the knowledge base."
                ),
            )

        # 7. Non-medical question
        return ValidationResult(
            category=QuestionCategory.UNRELATED,
            action=SafetyAction.REFUSE,
            message=(
                "This question is outside the scope of the medical "
                "knowledge assistant."
            ),
        )

    def _contains_medical_keyword(self, question: str) -> bool:
        words = set(re.findall(r"\b[a-zA-Z]+\b", question))

        if words.intersection(self.MEDICAL_KEYWORDS):
            return True

        return self._contains_supported_topic(question)

    def _contains_supported_topic(self, question: str) -> bool:
        return any(
            topic in question
            for topic in self.SUPPORTED_TOPICS
        )

    @staticmethod
    def _contains_any(text: str, patterns: list[str]) -> bool:
        return any(pattern in text for pattern in patterns)

    @staticmethod
    def _matches(question: str, patterns: list[str]) -> bool:
        return any(
            re.search(pattern, question)
            for pattern in patterns
        )