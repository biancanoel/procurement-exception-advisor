"""Tests for the emergency procurement decision criteria."""
# These will catch duplicate IDs, broken definitions, and lookup errors before building the agent.
import pytest

from decision.emergency_criteria import (
    get_audit_readiness_criteria,
    get_audit_readiness_criterion,
    get_audit_readiness_criteria_count,
    get_emergency_criteria,
    get_emergency_criterion,
    get_required_emergency_criteria,
    get_emergency_criteria_count,
    get_procurement_criteria,
    get_procurement_criterion,
    get_procurement_criteria_count,
)
from models.criteria import EmergencyCriterion


EXPECTED_EMERGENCY_CRITERION_IDS = [
    "unexpected_event",
    "immediate_harm",
    "competition_impracticable",
]

EXPECTED_AUDIT_CRITERION_IDS = [
    "purchase_classification",
    "threshold_and_funding",
    "appropriate_response_scope",
    "vendor_selection",
    "price_reasonableness",
    "approval_authority",
    "remaining_compliance_requirements",
    "post_facto_formalization",
]


def test_returns_all_emergency_criteria_in_order() -> None:
    criteria = get_emergency_criteria()

    assert len(criteria) == get_emergency_criteria_count()

    assert [
        criterion.criterion_id
        for criterion in criteria
    ] == EXPECTED_EMERGENCY_CRITERION_IDS


def test_returns_all_audit_readiness_criteria_in_order() -> None:
    criteria = get_audit_readiness_criteria()

    assert len(criteria) == get_audit_readiness_criteria_count() == 8
    assert [
        criterion.criterion_id
        for criterion in criteria
    ] == EXPECTED_AUDIT_CRITERION_IDS


def test_procurement_criteria_partition_covers_all_eleven() -> None:
    emergency_ids = {
        criterion.criterion_id for criterion in get_emergency_criteria()
    }
    audit_ids = {
        criterion.criterion_id for criterion in get_audit_readiness_criteria()
    }
    all_ids = [
        criterion.criterion_id for criterion in get_procurement_criteria()
    ]

    assert emergency_ids.isdisjoint(audit_ids)
    assert get_procurement_criteria_count() == 11
    assert all_ids == (
        EXPECTED_EMERGENCY_CRITERION_IDS + EXPECTED_AUDIT_CRITERION_IDS
    )


def test_all_criteria_are_valid_models() -> None:
    criteria = get_procurement_criteria()

    assert all(
        isinstance(criterion, EmergencyCriterion)
        for criterion in criteria
    )


def test_criterion_ids_are_unique() -> None:
    criteria = get_procurement_criteria()

    criterion_ids = [
        criterion.criterion_id
        for criterion in criteria
    ]

    assert len(criterion_ids) == len(set(criterion_ids))


def test_each_criterion_has_questions_and_risk() -> None:
    for criterion in get_procurement_criteria():
        assert criterion.questions_to_answer
        assert criterion.risk_if_missing
        assert criterion.name
        assert criterion.description


def test_get_emergency_criterion_returns_match() -> None:
    criterion = get_emergency_criterion(
        "competition_impracticable"
    )

    assert criterion.name == "Competition is impracticable"


def test_get_emergency_criterion_rejects_unknown_id() -> None:
    with pytest.raises(
        KeyError,
        match="Unknown emergency criterion",
    ):
        get_emergency_criterion("does_not_exist")


def test_stage_specific_lookup_rejects_criterion_from_other_stage() -> None:
    with pytest.raises(KeyError, match="Unknown emergency criterion"):
        get_emergency_criterion("approval_authority")

    assert (
        get_audit_readiness_criterion("approval_authority").name
        == "Required authority and approvals"
    )
    assert (
        get_procurement_criterion("approval_authority").name
        == "Required authority and approvals"
    )


def test_required_criteria_are_subset_of_all_criteria() -> None:
    all_ids = {
        criterion.criterion_id
        for criterion in get_emergency_criteria()
    }

    required_ids = {
        criterion.criterion_id
        for criterion in get_required_emergency_criteria()
    }

    assert required_ids
    assert required_ids <= all_ids


def test_approval_authority_does_not_allow_partial_support() -> None:
    criterion = get_audit_readiness_criterion(
        "approval_authority"
    )

    assert criterion.allows_partial_support is False


def test_appropriate_response_scope_replaces_separate_response_criteria() -> None:
    criterion = get_audit_readiness_criterion(
        "appropriate_response_scope"
    )

    assert criterion.description == (
        "Determine whether the proposed purchase directly addresses the "
        "emergency and whether its scope, quantity, value, and duration are "
        "reasonably limited to what is necessary to prevent, reduce, "
        "stabilize, or resolve the identified harm."
    )
    with pytest.raises(KeyError):
        get_procurement_criterion("necessary_response")
    with pytest.raises(KeyError):
        get_procurement_criterion("limited_scope")
