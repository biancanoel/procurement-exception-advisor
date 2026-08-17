"""Validated data models for emergency-procurement evaluation cases."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base model that rejects unexpected fields and trims strings."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class Jurisdiction(StrictModel):
    state: str = Field(min_length=1)
    agency: str = Field(min_length=1)


class AvailableDocument(StrictModel):
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class EmergencyCaseInput(StrictModel):
    schema_version: Literal["1.0"]
    case_id: str = Field(pattern=r"^EM-[0-9]{3}$")
    title: str = Field(min_length=1)
    workflow_type: Literal["emergency_procurement"]
    jurisdiction: Jurisdiction
    department: str = Field(min_length=1)
    estimated_amount_usd: float = Field(ge=0)
    proposed_vendor: str = Field(min_length=1)
    request_text: str = Field(min_length=1)
    available_documents: list[AvailableDocument]

    @field_validator("available_documents")
    @classmethod
    def document_ids_must_be_unique(
        cls,
        documents: list[AvailableDocument],
    ) -> list[AvailableDocument]:
        document_ids = [document.document_id for document in documents]

        if len(document_ids) != len(set(document_ids)):
            raise ValueError(
                "available_documents contains duplicate document_id values"
            )

        return documents

    @property
    def available_document_ids(self) -> set[str]:
        """Return the document IDs available to the agent for this case."""

        return {
            document.document_id
            for document in self.available_documents
        }


class RecommendationCode(StrEnum):
    SUFFICIENTLY_SUPPORTED = "sufficiently_supported"
    ADDITIONAL_EVIDENCE_REQUIRED = "additional_evidence_required"
    NOT_SUFFICIENTLY_SUPPORTED = "not_sufficiently_supported"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


class EvidenceStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"
    CONTRADICTED = "contradicted"
    NOT_APPLICABLE = "not_applicable"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


class ExpectedRecommendation(StrictModel):
    code: RecommendationCode
    summary: str = Field(min_length=1)


class CriterionAssessment(StrictModel):
    """This is the model for expected outcome in the mock case dataset. This is 
    not the application output model (see models.assessment.CriterionResult) """
    criterion_id: str = Field(min_length=1)
    criterion: str = Field(min_length=1)
    status: EvidenceStatus
    rationale: str = Field(min_length=1)
    evidence_document_ids: list[str]

    @field_validator("evidence_document_ids")
    @classmethod
    def evidence_ids_must_be_unique(
        cls,
        values: list[str],
    ) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError(
                "evidence_document_ids contains duplicate values"
            )

        return values


class EmergencyCaseExpected(StrictModel):
    schema_version: Literal["1.0"]
    case_id: str = Field(pattern=r"^EM-[0-9]{3}$")
    expected_classification: str = Field(min_length=1)
    expected_recommendation: ExpectedRecommendation
    criteria_assessment: list[CriterionAssessment] = Field(min_length=1)
    expected_follow_up_questions: list[str]
    expected_next_steps: list[str]
    critical_agent_behaviors: list[str]
    failure_modes_to_avoid: list[str]

    @field_validator("criteria_assessment")
    @classmethod
    def criterion_ids_must_be_unique(
        cls,
        assessments: list[CriterionAssessment],
    ) -> list[CriterionAssessment]:
        criterion_ids = [
            assessment.criterion_id
            for assessment in assessments
        ]

        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError(
                "criteria_assessment contains duplicate criterion_id values"
            )

        return assessments


class CaseManifestEntry(StrictModel):
    case_id: str = Field(pattern=r"^EM-[0-9]{3}$")
    title: str = Field(min_length=1)
    input_file: str = Field(min_length=1)
    expected_file: str = Field(min_length=1)
    primary_capability_tested: str = Field(min_length=1)


class CaseManifest(StrictModel):
    schema_version: Literal["1.0"]
    dataset_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    usage_note: str = Field(min_length=1)
    cases: list[CaseManifestEntry] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def case_ids_must_be_unique(
        cls,
        cases: list[CaseManifestEntry],
    ) -> list[CaseManifestEntry]:
        case_ids = [case.case_id for case in cases]

        if len(case_ids) != len(set(case_ids)):
            raise ValueError(
                "manifest contains duplicate case_id values"
            )

        return cases