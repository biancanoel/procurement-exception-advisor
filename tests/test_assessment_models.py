"""Tests for emergency assessment output models."""

import pytest
from pydantic import ValidationError

from models.assessment import (
    CriterionResult,
    EmergencyAssessment,
    FinalRecommendation,
)
from models.criteria import CriterionStatus


def test_criterion_result_accepts_valid_data() -> None:
    result = CriterionResult(
        criterion_id="immediate_harm",
        status=CriterionStatus.SUPPORTED,
        rationale=(
            "The incident report documents an immediate public safety risk."
        ),
        confidence=0.95,
    )

    assert result.criterion_id == "immediate_harm"
    assert result.status == CriterionStatus.SUPPORTED


def test_criterion_result_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        CriterionResult(
            criterion_id="immediate_harm",
            status=CriterionStatus.SUPPORTED,
            rationale="Supported by the incident report.",
            confidence=1.5,
        )


def test_assessment_accepts_valid_data() -> None:
    assessment = EmergencyAssessment(
        case_id="EM-001",
        recommendation=(
            FinalRecommendation.SUFFICIENTLY_SUPPORTED
        ),
        executive_summary=(
            "The emergency is supported, subject to approval verification."
        ),
        classification="Emergency public project",
        criterion_results=[
            CriterionResult(
                criterion_id="immediate_harm",
                status=CriterionStatus.SUPPORTED,
                rationale="Wastewater may enter a storm drain.",
                confidence=0.98,
            )
        ],
        overall_confidence=0.90,
    )

    assert assessment.case_id == "EM-001"
    assert len(assessment.criterion_results) == 1


def test_assessment_rejects_duplicate_criterion_ids() -> None:
    duplicate_result = CriterionResult(
        criterion_id="immediate_harm",
        status=CriterionStatus.SUPPORTED,
        rationale="Supported.",
        confidence=0.9,
    )

    with pytest.raises(
        ValidationError,
        match="duplicate criterion IDs",
    ):
        EmergencyAssessment(
            case_id="EM-001",
            recommendation=(
                FinalRecommendation.SUFFICIENTLY_SUPPORTED
            ),
            executive_summary="Valid assessment.",
            classification="Emergency public project",
            criterion_results=[
                duplicate_result,
                duplicate_result,
            ],
            overall_confidence=0.9,
        )