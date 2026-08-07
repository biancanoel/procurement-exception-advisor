"""Tests for the emergency procurement decision criteria."""
# These will catch duplicate IDs, broken definitions, and lookup errors before building the agent.
import pytest

from decision.emergency_criteria import (
    get_emergency_criteria,
    get_emergency_criterion,
    get_required_emergency_criteria,
    get_emergency_criteria_count
)
from models.criteria import EmergencyCriterion


EXPECTED_CRITERION_IDS = [
    "purchase_classification",
    "threshold_and_funding",
    "unexpected_event",
    "immediate_harm",
    "competition_impracticable",
    "necessary_response",
    "limited_scope",
    "vendor_selection",
    "price_reasonableness",
    "approval_authority",
    "remaining_compliance_requirements",
    "documentation_complete",
    "post_facto_formalization",
]


def test_returns_all_emergency_criteria_in_order() -> None:
    criteria = get_emergency_criteria()

    assert len(criteria) == get_emergency_criteria_count()

    assert [
        criterion.criterion_id
        for criterion in criteria
    ] == EXPECTED_CRITERION_IDS


def test_all_criteria_are_valid_models() -> None:
    criteria = get_emergency_criteria()

    assert all(
        isinstance(criterion, EmergencyCriterion)
        for criterion in criteria
    )


def test_criterion_ids_are_unique() -> None:
    criteria = get_emergency_criteria()

    criterion_ids = [
        criterion.criterion_id
        for criterion in criteria
    ]

    assert len(criterion_ids) == len(set(criterion_ids))


def test_each_criterion_has_questions_and_risk() -> None:
    for criterion in get_emergency_criteria():
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
    criterion = get_emergency_criterion(
        "approval_authority"
    )

    assert criterion.allows_partial_support is False