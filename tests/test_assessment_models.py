"""Tests for emergency assessment output models."""

import pytest
from pydantic import ValidationError

from models.assessment import (
    CriterionResult,
    EmergencyAssessment,
    EvidenceReference,
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


@pytest.mark.parametrize(
    "gap_fields",
    [
        {
            "conflicting_evidence": [
                EvidenceReference(
                    source_id="DOC-001",
                    source_type="case_document",
                    description="The report contains a conflicting date.",
                )
            ]
        },
        {
            "requires_human_review": True,
            "human_review_reason": "Approval must be confirmed.",
        },
        {"human_review_reason": "Approval must be confirmed."},
    ],
)
def test_supported_result_rejects_unresolved_gap_fields(
    gap_fields: dict,
) -> None:
    with pytest.raises(
        ValidationError,
        match="supported criterion cannot contain material unresolved fields",
    ):
        CriterionResult(
            criterion_id="immediate_harm",
            status=CriterionStatus.SUPPORTED,
            rationale="The available evidence supports immediate harm.",
            confidence=0.9,
            **gap_fields,
        )


def test_supported_result_can_keep_nonmaterial_documentation_items() -> None:
    result = CriterionResult(
        criterion_id="immediate_harm",
        status=CriterionStatus.SUPPORTED,
        rationale=(
            "Current operational evidence establishes immediate harm; the "
            "remaining items would only improve the audit record."
        ),
        missing_evidence=["Supplemental incident chronology"],
        follow_up_questions=["Can the chronology be added to the file?"],
        confidence=0.9,
    )

    assert result.status == CriterionStatus.SUPPORTED
    assert result.missing_evidence == ["Supplemental incident chronology"]


def test_partially_supported_result_can_preserve_evidence_gaps() -> None:
    result = CriterionResult(
        criterion_id="immediate_harm",
        status=CriterionStatus.PARTIALLY_SUPPORTED,
        rationale="Some evidence supports harm, but timing is unclear.",
        missing_evidence=["Incident timeline"],
        follow_up_questions=["When would the harm occur?"],
        confidence=0.5,
    )

    assert result.missing_evidence == ["Incident timeline"]


def test_unknown_funding_cannot_be_not_supported_from_missing_evidence() -> None:
    with pytest.raises(
        ValidationError,
        match="requires affirmative adverse evidence",
    ):
        CriterionResult(
            criterion_id="threshold_and_funding",
            status=CriterionStatus.NOT_SUPPORTED,
            rationale="The funding source and applicable threshold are unknown.",
            missing_evidence=[
                "Funding source",
                "Applicable purchasing threshold",
            ],
            confidence=0.2,
        )


def test_affirmative_adverse_evidence_can_support_negative_finding() -> None:
    result = CriterionResult(
        criterion_id="competition_impracticable",
        status=CriterionStatus.NOT_SUPPORTED,
        rationale=(
            "A qualified alternate vendor documented that it can deliver "
            "within the required timeframe."
        ),
        supporting_evidence=[
            EvidenceReference(
                source_id="EM005-D07",
                source_type="case_document",
                description=(
                    "The alternate supplier can deliver before existing "
                    "inventory is exhausted."
                ),
            )
        ],
        missing_evidence=["Written procurement file determination"],
        confidence=0.85,
    )

    assert result.status == CriterionStatus.NOT_SUPPORTED
    assert result.missing_evidence == ["Written procurement file determination"]


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
