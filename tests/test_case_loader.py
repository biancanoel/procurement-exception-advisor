"""Integration tests for the JSON case loader."""

import pytest

from data.case_loader import (
    CaseFileNotFoundError,
    iter_case_pairs,
    list_case_ids,
    load_case,
    load_case_pair,
)


def test_manifest_lists_all_five_cases() -> None:
    assert list_case_ids() == [
        "EM-001",
        "EM-002",
        "EM-003",
        "EM-004",
        "EM-005",
    ]


def test_load_case_accepts_flexible_case_id_format() -> None:
    case = load_case("em-001")

    assert case.case_id == "EM-001"
    assert case.department == "Public Works"

    assert (
        case.proposed_vendor
        == "Inland Utility Contractors"
    )


def test_all_case_pairs_load_and_cross_validate() -> None:
    pairs = list(iter_case_pairs())

    assert len(pairs) == 5

    for case, expected in pairs:
        assert case.case_id == expected.case_id
        assert expected.criteria_assessment


def test_expected_evidence_ids_exist_in_input_case() -> None:
    case, expected = load_case_pair("EM-005")

    evidence_ids = {
        document_id
        for assessment in expected.criteria_assessment
        for document_id in assessment.evidence_document_ids
    }

    assert evidence_ids <= case.available_document_ids


def test_unknown_case_raises_clear_error() -> None:
    with pytest.raises(
        CaseFileNotFoundError,
        match="not found",
    ):
        load_case("EM-999")