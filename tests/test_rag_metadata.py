"""Tests for procurement-document retrieval metadata."""

from datetime import date

import pytest
from pydantic import ValidationError

from rag import ProcurementDocumentMetadata


def make_metadata(**overrides: object) -> dict[str, object]:
    """Create valid procurement-policy metadata for model tests."""

    data: dict[str, object] = {
        "document_id": "CA-PCC-22050",
        "title": "California Public Contract Code section 22050",
        "jurisdiction": "California",
        "agency": "State of California",
        "document_type": "statute",
        "effective_date": "2025-01-01",
        "authority_level": "state_statute",
        "exception_type": "emergency",
        "section": "22050",
        "page": 1,
        "source_path": "policies/california/pcc-22050.pdf",
    }
    data.update(overrides)
    return data


def test_metadata_accepts_and_normalizes_valid_data() -> None:
    metadata = ProcurementDocumentMetadata.model_validate(make_metadata())

    assert metadata.document_id == "CA-PCC-22050"
    assert metadata.effective_date == date(2025, 1, 1)
    assert metadata.page == 1


def test_metadata_supports_document_level_records() -> None:
    metadata = ProcurementDocumentMetadata.model_validate(
        make_metadata(
            effective_date=None,
            exception_type=None,
            section=None,
            page=None,
        )
    )

    assert metadata.effective_date is None
    assert metadata.section is None
    assert metadata.page is None


def test_metadata_preserves_optional_subject() -> None:
    metadata = ProcurementDocumentMetadata.model_validate(
        make_metadata(
            subject="procurement classification and solicitation thresholds"
        )
    )

    assert metadata.subject == (
        "procurement classification and solicitation thresholds"
    )


def test_metadata_rejects_non_positive_page() -> None:
    with pytest.raises(ValidationError):
        ProcurementDocumentMetadata.model_validate(make_metadata(page=0))


def test_metadata_rejects_reversed_page_span() -> None:
    with pytest.raises(ValidationError, match="page_end cannot precede page"):
        ProcurementDocumentMetadata.model_validate(
            make_metadata(page=4, page_end=3)
        )


def test_metadata_rejects_blank_required_fields() -> None:
    with pytest.raises(ValidationError):
        ProcurementDocumentMetadata.model_validate(make_metadata(title="   "))


def test_metadata_rejects_unexpected_fields() -> None:
    with pytest.raises(ValidationError):
        ProcurementDocumentMetadata.model_validate(
            make_metadata(embedding=[0.1, 0.2])
        )
