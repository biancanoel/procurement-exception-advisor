"""Unit tests for emergency-case Pydantic models."""

import pytest
from pydantic import ValidationError

from models.cases import (
    AvailableDocument,
    EmergencyCaseInput,
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