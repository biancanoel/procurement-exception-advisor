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


class ProcurementContext(StrictModel):
    """Procurement baseline established before audit-readiness assessment."""

    case_id: str = Field(pattern=r"^EM-[0-9]{3}$")
    purchase_classification: str | None = None
    estimated_purchase_value_usd: float | None = Field(default=None, ge=0)
    funding_source: str | None = None
    applicable_threshold: str | None = None
    normal_procurement_method: str | None = None
    normal_approval_authority: list[str] | None = None
    special_procurement_requirements: list[str] | None = None
    requirements_modified_by_emergency: list[str] | None = None
    requirements_still_applicable: list[str] | None = None
    unresolved_questions: list[str] = Field(default_factory=list)
    requires_human_input: bool = False
    sources_used: list[EvidenceReference] = Field(default_factory=list)

    @field_validator(
        "normal_approval_authority",
        "special_procurement_requirements",
        "requirements_modified_by_emergency",
        "requirements_still_applicable",
        "unresolved_questions",
    )
    @classmethod
    def context_list_values_must_be_unique(
        cls,
        values: list[str] | None,
    ) -> list[str] | None:
        """Reject repeated contextual findings or follow-up questions."""

        if values is None:
            return values
        normalized = [value.casefold() for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError("List values must be unique.")
        return values

    @model_validator(mode="after")
    def unknown_context_requires_questions(self) -> ProcurementContext:
        """Keep unknown fields explicit and tied to human follow-up."""

        contextual_values = (
            self.purchase_classification,
            self.estimated_purchase_value_usd,
            self.funding_source,
            self.applicable_threshold,
            self.normal_procurement_method,
            self.normal_approval_authority,
            self.special_procurement_requirements,
            self.requirements_modified_by_emergency,
            self.requirements_still_applicable,
        )
        has_unknowns = any(value is None for value in contextual_values)
        if has_unknowns and not self.unresolved_questions:
            raise ValueError(
                "Unknown procurement context requires unresolved_questions."
            )
        if self.requires_human_input != bool(self.unresolved_questions):
            raise ValueError(
                "requires_human_input must match whether unresolved_questions "
                "are present."
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


class AuditReadinessAssessment(StrictModel):
    """Six-criterion review of an emergency procurement file's readiness."""

    schema_version: str = "1.0"

    case_id: str = Field(
        pattern=r"^EM-[0-9]{3}$"
    )

    recommendation: FinalRecommendation

    executive_summary: str = Field(min_length=1)

    classification: str = Field(min_length=1)

    criterion_results: list[CriterionResult] = Field(
        min_length=6,
        max_length=6,
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


class AuditReadinessCriterionReassessment(StrictModel):
    """Targeted updates for criteria that remained unresolved after research."""

    case_id: str = Field(pattern=r"^EM-[0-9]{3}$")

    criterion_results: list[CriterionResult] = Field(
        min_length=1,
        max_length=6,
    )

    @field_validator("criterion_results")
    @classmethod
    def criterion_ids_must_be_unique(
        cls,
        results: list[CriterionResult],
    ) -> list[CriterionResult]:
        """Reject duplicate targeted updates before they are merged."""

        criterion_ids = [result.criterion_id for result in results]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError(
                "criterion_results contains duplicate criterion IDs."
            )
        return results


class EmergencyProcurementAssessment(StrictModel):
    """Complete result containing verification, context, and audit stages."""

    schema_version: str = "1.0"

    case_id: str = Field(pattern=r"^EM-[0-9]{3}$")

    emergency_verification: EmergencyVerification

    procurement_context: ProcurementContext | None = None

    audit_readiness: AuditReadinessAssessment | None = None

    @model_validator(mode="after")
    def stage_case_ids_and_routing_must_be_consistent(
        self,
    ) -> EmergencyProcurementAssessment:
        """Keep nested stage results aligned with the workflow outcome."""

        if self.emergency_verification.case_id != self.case_id:
            raise ValueError(
                "emergency_verification case_id must match assessment case_id."
            )
        if self.procurement_context is not None:
            if self.procurement_context.case_id != self.case_id:
                raise ValueError(
                    "procurement_context case_id must match assessment case_id."
                )
            if self.emergency_verification.emergency_is_verified is not True:
                raise ValueError(
                    "procurement_context requires a verified emergency."
                )
        if self.audit_readiness is not None:
            if self.audit_readiness.case_id != self.case_id:
                raise ValueError(
                    "audit_readiness case_id must match assessment case_id."
                )
            if self.emergency_verification.emergency_is_verified is not True:
                raise ValueError(
                    "audit_readiness requires a verified emergency."
                )
        return self

    @property
    def criterion_results(self) -> list[CriterionResult]:
        """Return available stage results in workflow order without copying state."""

        results = list(self.emergency_verification.criterion_results)
        if self.audit_readiness is not None:
            results.extend(self.audit_readiness.criterion_results)
        return results
