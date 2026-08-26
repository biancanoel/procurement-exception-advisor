"""Tests for emergency assessment output models."""

import pytest
from pydantic import ValidationError

from models.assessment import (
    AuditReadinessAssessment,
    CriterionResult,
    EmergencyProcurementAssessment,
    EmergencyVerification,
    EvidenceReference,
    FinalRecommendation,
)
from models.criteria import CriterionStatus


VERIFICATION_IDS = (
    "unexpected_event",
    "immediate_harm",
    "competition_impracticable",
)

AUDIT_IDS = (
    "appropriate_response_scope",
    "vendor_selection",
    "price_reasonableness",
    "approval_authority",
    "remaining_compliance_requirements",
    "post_facto_formalization",
)


def verification_result(
    criterion_id: str,
    status: CriterionStatus,
) -> CriterionResult:
    kwargs = {}
    if status == CriterionStatus.NOT_SUPPORTED:
        kwargs["supporting_evidence"] = [
            EvidenceReference(
                source_id="DOC-001",
                source_type="case_document",
                description="Affirmative evidence that the criterion fails.",
            )
        ]
    return CriterionResult(
        criterion_id=criterion_id,
        status=status,
        rationale=f"Rationale for {criterion_id}.",
        missing_evidence=(
            ["Material evidence"]
            if status == CriterionStatus.NOT_EVALUATED
            else []
        ),
        confidence=0.9 if status == CriterionStatus.SUPPORTED else 0.4,
        **kwargs,
    )


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


def test_emergency_verification_accepts_supported_yes_result() -> None:
    verification = EmergencyVerification(
        case_id="EM-001",
        emergency_is_verified=True,
        criterion_results=[
            verification_result(criterion_id, CriterionStatus.SUPPORTED)
            for criterion_id in VERIFICATION_IDS
        ],
        rationale="All three emergency elements are affirmatively supported.",
        confidence=0.9,
    )

    assert verification.emergency_is_verified is True
    assert len(verification.criterion_results) == 3


def test_emergency_verification_accepts_affirmative_no_result() -> None:
    results = [
        verification_result(criterion_id, CriterionStatus.SUPPORTED)
        for criterion_id in VERIFICATION_IDS
    ]
    results[-1] = verification_result(
        "competition_impracticable",
        CriterionStatus.NOT_SUPPORTED,
    )

    verification = EmergencyVerification(
        case_id="EM-001",
        emergency_is_verified=False,
        criterion_results=results,
        rationale="Competition remains practical, so no emergency exception exists.",
        confidence=0.85,
    )

    assert verification.emergency_is_verified is False


def test_emergency_verification_requires_unresolved_result_when_indeterminate() -> None:
    verification = EmergencyVerification(
        case_id="EM-001",
        emergency_is_verified=None,
        criterion_results=[
            verification_result(
                "unexpected_event",
                CriterionStatus.NOT_EVALUATED,
            ),
            verification_result("immediate_harm", CriterionStatus.SUPPORTED),
            verification_result(
                "competition_impracticable",
                CriterionStatus.SUPPORTED,
            ),
        ],
        rationale="The event evidence is incomplete.",
        confidence=0.4,
    )

    assert verification.emergency_is_verified is None


def test_emergency_verification_reconciles_yes_with_unresolved_result() -> None:
    verification = EmergencyVerification(
        case_id="EM-001",
        emergency_is_verified=True,
        criterion_results=[
            verification_result(
                "unexpected_event",
                CriterionStatus.NOT_EVALUATED,
            ),
            verification_result("immediate_harm", CriterionStatus.SUPPORTED),
            verification_result(
                "competition_impracticable",
                CriterionStatus.SUPPORTED,
            ),
        ],
        rationale="The record is incomplete.",
        confidence=0.4,
    )

    assert verification.emergency_is_verified is None


def test_emergency_verification_reconciles_false_without_adverse_result() -> None:
    verification = EmergencyVerification(
        case_id="EM-005",
        emergency_is_verified=False,
        criterion_results=[
            verification_result(
                "unexpected_event",
                CriterionStatus.PARTIALLY_SUPPORTED,
            ),
            verification_result("immediate_harm", CriterionStatus.SUPPORTED),
            verification_result(
                "competition_impracticable",
                CriterionStatus.PARTIALLY_SUPPORTED,
            ),
        ],
        rationale="Material evidence remains unresolved.",
        confidence=0.5,
    )

    assert verification.emergency_is_verified is None


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


def test_missing_evidence_alone_cannot_support_negative_finding() -> None:
    with pytest.raises(
        ValidationError,
        match="requires affirmative adverse evidence",
    ):
        CriterionResult(
            criterion_id="approval_authority",
            status=CriterionStatus.NOT_SUPPORTED,
            rationale="The applicable approval authority is unknown.",
            missing_evidence=[
                "Applicable approval authority",
                "Approval record",
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


def test_audit_readiness_assessment_accepts_six_results() -> None:
    assessment = AuditReadinessAssessment(
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
                criterion_id=criterion_id,
                status=CriterionStatus.SUPPORTED,
                rationale=f"The record supports {criterion_id}.",
                confidence=0.98,
            )
            for criterion_id in AUDIT_IDS
        ],
        overall_confidence=0.90,
    )

    assert assessment.case_id == "EM-001"
    assert len(assessment.criterion_results) == 6


def test_audit_readiness_rejects_duplicate_criterion_ids() -> None:
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
        AuditReadinessAssessment(
            case_id="EM-001",
            recommendation=(
                FinalRecommendation.SUFFICIENTLY_SUPPORTED
            ),
            executive_summary="Valid assessment.",
            classification="Emergency public project",
            criterion_results=[
                duplicate_result,
                duplicate_result,
                *[
                    CriterionResult(
                        criterion_id=criterion_id,
                        status=CriterionStatus.SUPPORTED,
                        rationale="Supported.",
                        confidence=0.9,
                    )
                    for criterion_id in AUDIT_IDS[:4]
                ],
            ],
            overall_confidence=0.9,
        )


def test_complete_assessment_nests_both_stage_results() -> None:
    verification = EmergencyVerification(
        case_id="EM-001",
        emergency_is_verified=True,
        criterion_results=[
            verification_result(criterion_id, CriterionStatus.SUPPORTED)
            for criterion_id in VERIFICATION_IDS
        ],
        rationale="The emergency is verified.",
        confidence=0.9,
    )
    audit = AuditReadinessAssessment(
        case_id="EM-001",
        recommendation=FinalRecommendation.SUFFICIENTLY_SUPPORTED,
        executive_summary="The file is audit-ready.",
        classification="Emergency procurement",
        criterion_results=[
            CriterionResult(
                criterion_id=criterion_id,
                status=CriterionStatus.SUPPORTED,
                rationale="Supported.",
                confidence=0.9,
            )
            for criterion_id in AUDIT_IDS
        ],
        overall_confidence=0.9,
    )

    assessment = EmergencyProcurementAssessment(
        case_id="EM-001",
        emergency_verification=verification,
        audit_readiness=audit,
    )

    assert assessment.emergency_verification is verification
    assert assessment.audit_readiness is audit
    assert len(assessment.criterion_results) == 9


def test_complete_assessment_rejects_audit_stage_without_verified_emergency() -> None:
    rejected_results = [
        verification_result(criterion_id, CriterionStatus.SUPPORTED)
        for criterion_id in VERIFICATION_IDS
    ]
    rejected_results[0] = verification_result(
        "unexpected_event",
        CriterionStatus.NOT_SUPPORTED,
    )
    rejected = EmergencyVerification(
        case_id="EM-001",
        emergency_is_verified=False,
        criterion_results=rejected_results,
        rationale="The event was foreseeable.",
        confidence=0.9,
    )
    audit = AuditReadinessAssessment(
        case_id="EM-001",
        recommendation=FinalRecommendation.SUFFICIENTLY_SUPPORTED,
        executive_summary="The file is audit-ready.",
        classification="Emergency procurement",
        criterion_results=[
            CriterionResult(
                criterion_id=criterion_id,
                status=CriterionStatus.SUPPORTED,
                rationale="Supported.",
                confidence=0.9,
            )
            for criterion_id in AUDIT_IDS
        ],
        overall_confidence=0.9,
    )

    with pytest.raises(ValidationError, match="requires a verified emergency"):
        EmergencyProcurementAssessment(
            case_id="EM-001",
            emergency_verification=rejected,
            audit_readiness=audit,
        )
