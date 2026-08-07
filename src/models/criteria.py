# Unexpected event
# Immediate harm if delayed
# Competition impracticable
# Proposed action necessary
# Scope limited to the emergency
# Vendor selection supported
# Price reasonableness
# Required approvals
# Documentation completeness
# Human review triggers

"""Models for emergency procurement evaluation criteria."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base model that rejects unexpected fields."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class CriterionStatus(StrEnum):
    """Possible assessment results for an emergency criterion."""

    NOT_EVALUATED = "not_evaluated"
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"
    CONTRADICTED = "contradicted"
    NOT_APPLICABLE = "not_applicable"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


class EvidenceType(StrEnum):
    """Common categories of supporting evidence."""

    USER_STATEMENT = "user_statement"
    INCIDENT_REPORT = "incident_report"
    TECHNICAL_ASSESSMENT = "technical_assessment"
    PHOTOGRAPH = "photograph"
    EMAIL = "email"
    QUOTE = "quote"
    PRICE_COMPARISON = "price_comparison"
    CONTRACT = "contract"
    POLICY = "policy"
    STATUTE = "statute"
    APPROVAL_RECORD = "approval_record"
    VENDOR_AVAILABILITY = "vendor_availability"
    MARKET_RESEARCH = "market_research"
    TIMELINE = "timeline"
    LICENSE_OR_REGISTRATION = "license_or_registration"
    INSURANCE_DOCUMENT = "insurance_document"
    OTHER = "other"


class EmergencyCriterion(BaseModel):
    """
    Defines one question the agent must evaluate for an emergency
    procurement request.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    criterion_id: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
    )

    name: str = Field(min_length=1)

    description: str = Field(min_length=1)

    questions_to_answer: list[str] = Field(
        min_length=1,
    )

    expected_evidence: list[str] = Field(
        default_factory=list,
    )

    preferred_evidence_types: list[EvidenceType] = Field(
        default_factory=list,
    )

    risk_if_missing: str = Field(min_length=1)

    required_for_recommendation: bool = True

    allows_partial_support: bool = True

    human_review_triggers: list[str] = Field(
        default_factory=list,
    )

    @field_validator(
        "questions_to_answer",
        "expected_evidence",
        "human_review_triggers",
    )
    @classmethod
    def list_values_must_be_unique(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized = [value.casefold() for value in values]

        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "List values must be unique."
            )

        return values


class CriterionEvidence(BaseModel):
    """Evidence associated with one criterion assessment."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    document_id: str | None = None

    evidence_type: EvidenceType

    description: str = Field(min_length=1)

    supports_criterion: bool

    source_location: str | None = None

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )


class CriterionAssessment(BaseModel):
    """The agent's assessment of one emergency criterion."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    criterion_id: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
    )

    status: CriterionStatus = (
        CriterionStatus.NOT_EVALUATED
    )

    rationale: str | None = None

    supporting_evidence: list[CriterionEvidence] = Field(
        default_factory=list,
    )

    missing_evidence: list[str] = Field(
        default_factory=list,
    )

    conflicting_evidence: list[str] = Field(
        default_factory=list,
    )

    follow_up_questions: list[str] = Field(
        default_factory=list,
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    requires_human_review: bool = False
