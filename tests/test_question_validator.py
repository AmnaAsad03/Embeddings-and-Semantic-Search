import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.safety.question_validator import (
    QuestionCategory,
    QuestionValidator,
    SafetyAction,
)


@pytest.fixture
def validator():
    return QuestionValidator()


def test_supported_medical_question(validator):
    result = validator.validate("What are the symptoms of asthma?")

    assert result.category == QuestionCategory.RELEVANT_MEDICAL
    assert result.action == SafetyAction.ALLOW


def test_supported_multiword_topic(validator):
    result = validator.validate("What is pulmonary hypertension?")

    assert result.category == QuestionCategory.RELEVANT_MEDICAL
    assert result.action == SafetyAction.ALLOW


def test_unrelated_question(validator):
    result = validator.validate("What is the capital of France?")

    assert result.category == QuestionCategory.UNRELATED
    assert result.action == SafetyAction.REFUSE


def test_unsupported_medical_question(validator):
    result = validator.validate("What are the symptoms of kidney disease?")

    assert result.category == QuestionCategory.UNSUPPORTED_MEDICAL
    assert result.action == SafetyAction.REFUSE


def test_inappropriate_drug_request(validator):
    result = validator.validate("How can I make a dangerous drug?")

    assert result.category == QuestionCategory.INAPPROPRIATE
    assert result.action == SafetyAction.REFUSE


def test_inappropriate_poison_request(validator):
    result = validator.validate("How can I make poison?")

    assert result.category == QuestionCategory.INAPPROPRIATE
    assert result.action == SafetyAction.REFUSE


def test_dangerous_self_medication(validator):
    result = validator.validate("Should I take medicine for my headache?")

    assert result.category == QuestionCategory.DANGEROUS_SELF_MEDICATION
    assert result.action == SafetyAction.REFUSE


def test_medication_dose_change(validator):
    result = validator.validate("Should I increase my medication dose?")

    assert result.category == QuestionCategory.DANGEROUS_SELF_MEDICATION
    assert result.action == SafetyAction.REFUSE


def test_emergency_breathing_problem(validator):
    result = validator.validate(
        "I can't breathe and have severe chest pain."
    )

    assert result.category == QuestionCategory.EMERGENCY
    assert result.action == SafetyAction.ESCALATE


def test_emergency_unconsciousness(validator):
    result = validator.validate("The person is unconscious.")

    assert result.category == QuestionCategory.EMERGENCY
    assert result.action == SafetyAction.ESCALATE


def test_empty_question(validator):
    result = validator.validate("")

    assert result.category == QuestionCategory.UNRELATED
    assert result.action == SafetyAction.REFUSE


def test_whitespace_question(validator):
    result = validator.validate("   ")

    assert result.category == QuestionCategory.UNRELATED
    assert result.action == SafetyAction.REFUSE


def test_supported_pneumonia_question(validator):
    result = validator.validate(
        "What are the common complications of pneumonia?"
    )

    assert result.category == QuestionCategory.RELEVANT_MEDICAL
    assert result.action == SafetyAction.ALLOW


def test_supported_depression_question(validator):
    result = validator.validate("What are symptoms of depression?")

    assert result.category == QuestionCategory.RELEVANT_MEDICAL
    assert result.action == SafetyAction.ALLOW


def test_unsupported_cancer_question(validator):
    result = validator.validate("What are the symptoms of cancer?")

    assert result.category == QuestionCategory.UNSUPPORTED_MEDICAL
    assert result.action == SafetyAction.REFUSE