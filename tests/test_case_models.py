"""Unit tests for emergency-case Pydantic models."""

import pytest
from pydantic import ValidationError

from models.cases import (
    AvailableDocument,
    EmergencyCaseInput,
    EvidenceDocument,
    Jurisdiction,
)


def make_case(**overrides: object) -> dict[str, object]:
    """Create a valid case dictionary for model testing."""

    data: dict[str, object] = {
        "schema_version": "1.0",
        "case_id": "EM-999",
        "title": "Test Emergency",
        "workflow_type": "emergency_procurement",
        "jurisdiction": {
            "state": "California",
            "agency": "Pilot City",
        },
        "department": "Public Works",
        "estimated_amount_usd": 1000,
        "proposed_vendor": "Example Vendor",
        "request_text": (
            "An unexpected failure requires immediate review."
        ),
        "available_documents": [
            {
                "document_id": "TEST-D01",
                "title": "Incident Report",
                "summary": "Describes the incident.",
            }
        ],
    }

    data.update(overrides)

    return data


def test_input_model_accepts_valid_case() -> None:
    case = EmergencyCaseInput.model_validate(
        make_case()
    )

    assert case.case_id == "EM-999"
    assert case.estimated_amount_usd == 1000
    assert case.available_document_ids == {"TEST-D01"}

    assert isinstance(
        case.jurisdiction,
        Jurisdiction,
    )

    assert isinstance(
        case.available_documents[0],
        AvailableDocument,
    )


def test_input_model_requires_only_an_emergency_description() -> None:
    case = EmergencyCaseInput.model_validate(
        {
            "description": (
                "The water treatment plant has less than four days of "
                "disinfectant remaining."
            )
        }
    )

    assert case.description == case.request_text
    assert case.case_id == "EM-000"
    assert case.schema_version == "1.0"
    assert case.workflow_type == "emergency_procurement"
    assert case.title is None
    assert case.jurisdiction is None
    assert case.department is None
    assert case.estimated_amount_usd is None
    assert case.proposed_vendor is None
    assert case.available_documents == []
    assert case.case_evidence == []
    assert EmergencyCaseInput.model_json_schema()["required"] == [
        "description"
    ]


def test_input_model_preserves_uploaded_case_evidence() -> None:
    case = EmergencyCaseInput(
        description="A pump failed.",
        case_evidence=[
            EvidenceDocument(
                evidence_id="CASE-D01",
                filename="timeline.txt",
                extracted_text="The outage began at noon.",
            )
        ],
    )

    assert case.case_evidence[0].filename == "timeline.txt"
    assert case.case_evidence[0].file_type == "txt"
    assert case.case_evidence[0].extracted_text == "The outage began at noon."
    assert case.evidence_source_ids == {"CASE-D01"}


def test_input_model_rejects_missing_description() -> None:
    with pytest.raises(ValidationError, match="description"):
        EmergencyCaseInput.model_validate({})


def test_input_model_rejects_unexpected_fields() -> None:
    with pytest.raises(ValidationError):
        EmergencyCaseInput.model_validate(
            make_case(
                unexpected_field="not allowed"
            )
        )


def test_input_model_rejects_duplicate_document_ids() -> None:
    documents = [
        {
            "document_id": "TEST-D01",
            "title": "First document",
            "summary": "First summary",
        },
        {
            "document_id": "TEST-D01",
            "title": "Second document",
            "summary": "Second summary",
        },
    ]

    with pytest.raises(
        ValidationError,
        match="duplicate document_id",
    ):
        EmergencyCaseInput.model_validate(
            make_case(
                available_documents=documents
            )
        )
