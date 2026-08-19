"""Models for structured emergency procurement assessments."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from models.criteria import CriterionStatus


class StrictModel(BaseModel):
    """Base model that rejects unexpected fields."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class FinalRecommendation(StrEnum):
    """Possible overall recommendations for an emergency request."""

    SUFFICIENTLY_SUPPORTED = "sufficiently_supported"
    ADDITIONAL_EVIDENCE_REQUIRED = "additional_evidence_required"
    NOT_SUFFICIENTLY_SUPPORTED = "not_sufficiently_supported"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


class EvidenceReference(StrictModel):
    """A specific fact or document used in an assessment."""

    source_id: str = Field(min_length=1)

    source_type: str = Field(min_length=1)

    description: str = Field(min_length=1)

    source_location: str | None = None

    quote_or_fact: str | None = None


class CriterionResult(StrictModel):
    """The completed assessment for one emergency criterion.
      Stores the agen't conclusion for each checklist item
      This is the active application assessment model"""

    criterion_id: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
    )

    status: CriterionStatus

    rationale: str = Field(min_length=1)

    supporting_evidence: list[EvidenceReference] = Field(
        default_factory=list,
    )

    conflicting_evidence: list[EvidenceReference] = Field(
        default_factory=list,
    )

    missing_evidence: list[str] = Field(
        default_factory=list,
    )

    follow_up_questions: list[str] = Field(
        default_factory=list,
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    requires_human_review: bool = False

    human_review_reason: str | None = None

    @field_validator(
        "missing_evidence",
        "follow_up_questions",
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

    @model_validator(mode="after")
    def resolved_status_must_match_evidence(
        self,
    ) -> CriterionResult:
        """Keep resolved statuses consistent with their evidentiary basis."""

        if self.status == CriterionStatus.SUPPORTED:
            unresolved_fields: list[str] = []
            if self.conflicting_evidence:
                unresolved_fields.append("conflicting_evidence")
            if self.requires_human_review:
                unresolved_fields.append("requires_human_review")
            if self.human_review_reason:
                unresolved_fields.append("human_review_reason")

            if not unresolved_fields:
                return self
            fields = ", ".join(unresolved_fields)
            raise ValueError(
                "A supported criterion cannot contain material unresolved "
                f"fields: {fields}. Use an unresolved status or clear them."
            )

        if (
            self.status == CriterionStatus.NOT_SUPPORTED
            and not self.supporting_evidence
            and not self.conflicting_evidence
        ):
            raise ValueError(
                "A not-supported criterion requires affirmative adverse "
                "evidence. Missing evidence alone requires an unresolved "
                "status."
            )
        return self


class EmergencyVerification(StrictModel):
    """Gate determination for whether an emergency situation exists."""

    case_id: str = Field(pattern=r"^EM-[0-9]{3}$")

    emergency_is_verified: bool | None = Field(
        description=(
            "True only when all three criteria are supported; false only "
            "when at least one criterion has an affirmative adverse result; "
            "null when any material criterion remains unresolved without an "
            "affirmative adverse result."
        )
    )

    criterion_results: list[CriterionResult] = Field(
        min_length=3,
        max_length=3,
    )

    rationale: str = Field(min_length=1)

    confidence: float = Field(ge=0.0, le=1.0)

    source_ids_used: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def reconcile_determination_with_criterion_results(
        self,
    ) -> EmergencyVerification:
        """Derive the gate result from the model's criterion-level judgments."""

        criterion_ids = [
            result.criterion_id
            for result in self.criterion_results
        ]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError(
                "criterion_results contains duplicate criterion IDs."
            )

        adverse_statuses = {
            CriterionStatus.NOT_SUPPORTED,
            CriterionStatus.CONTRADICTED,
        }
        statuses = {
            result.status
            for result in self.criterion_results
        }

        if statuses.intersection(adverse_statuses):
            determination: bool | None = False
        elif all(
            result.status == CriterionStatus.SUPPORTED
            for result in self.criterion_results
        ):
            determination = True
        else:
            determination = None

        # Pydantic validates assignments on this model. Set the reconciled
        # summary directly to avoid recursively running this model validator.
        object.__setattr__(
            self,
            "emergency_is_verified",
            determination,
        )
        return self


class AuditRisk(StrictModel):
    """A risk that may weaken the procurement file."""

    risk_id: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*$",
    )

    title: str = Field(min_length=1)

    description: str = Field(min_length=1)

    severity: str = Field(
        pattern=r"^(low|medium|high|critical)$"
    )

    related_criterion_ids: list[str] = Field(
        default_factory=list,
    )

    recommended_action: str = Field(min_length=1)


class EmergencyAssessment(StrictModel):
    """Combines all criterion results into the final recommendation resulting in one 
    complete structured assessment for one emergency request."""

    schema_version: str = "1.0"

    case_id: str = Field(
        pattern=r"^EM-[0-9]{3}$"
    )

    recommendation: FinalRecommendation

    executive_summary: str = Field(min_length=1)

    classification: str = Field(min_length=1)

    criterion_results: list[CriterionResult] = Field(
        min_length=1,
    )

    audit_risks: list[AuditRisk] = Field(
        default_factory=list,
    )

    missing_documents: list[str] = Field(
        default_factory=list,
    )

    next_steps: list[str] = Field(
        default_factory=list,
    )

    required_approvals: list[str] = Field(
        default_factory=list,
    )

    source_ids_used: list[str] = Field(
        default_factory=list,
    )

    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    requires_human_review: bool = False

    human_review_reason: str | None = None

    @field_validator("criterion_results")
    @classmethod
    def criterion_ids_must_be_unique(
        cls,
        results: list[CriterionResult],
    ) -> list[CriterionResult]:
        criterion_ids = [
            result.criterion_id
            for result in results
        ]

        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError(
                "criterion_results contains duplicate criterion IDs."
            )

        return results
