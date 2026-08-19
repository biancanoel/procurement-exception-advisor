"""Tests for the initial emergency assessment builder."""

from data.case_loader import load_case
from decision.assessment_builder import create_initial_assessment
from decision.emergency_criteria import get_procurement_criteria
from models.assessment import FinalRecommendation
from models.criteria import CriterionStatus


def test_builder_preserves_case_id() -> None:
    case = load_case("EM-001")

    assessment = create_initial_assessment(case)

    assert assessment.case_id == "EM-001"


def test_builder_creates_result_for_every_criterion() -> None:
    case = load_case("EM-001")

    assessment = create_initial_assessment(case)
    criteria = get_procurement_criteria()

    assert len(assessment.criterion_results) == len(criteria)


def test_builder_preserves_criterion_order() -> None:
    case = load_case("EM-001")

    assessment = create_initial_assessment(case)

    expected_ids = [
        criterion.criterion_id
        for criterion in get_procurement_criteria()
    ]

    actual_ids = [
        result.criterion_id
        for result in assessment.criterion_results
    ]

    assert actual_ids == expected_ids


def test_all_criteria_start_not_evaluated() -> None:
    case = load_case("EM-001")

    assessment = create_initial_assessment(case)

    assert all(
        result.status == CriterionStatus.NOT_EVALUATED
        for result in assessment.criterion_results
    )


def test_all_criteria_start_with_zero_confidence() -> None:
    case = load_case("EM-001")

    assessment = create_initial_assessment(case)

    assert all(
        result.confidence == 0.0
        for result in assessment.criterion_results
    )


def test_initial_assessment_is_not_treated_as_final() -> None:
    case = load_case("EM-001")

    assessment = create_initial_assessment(case)

    assert (
        assessment.recommendation
        == FinalRecommendation.HUMAN_REVIEW_REQUIRED
    )

    assert assessment.requires_human_review is True
    assert assessment.overall_confidence == 0.0


def test_initial_assessment_has_placeholder_classification() -> None:
    case = load_case("EM-001")

    assessment = create_initial_assessment(case)

    assert assessment.classification == "Not yet classified"


def test_builder_works_for_every_mock_case() -> None:
    for case_id in [
        "EM-001",
        "EM-002",
        "EM-003",
        "EM-004",
        "EM-005",
    ]:
        case = load_case(case_id)
        assessment = create_initial_assessment(case)

        assert assessment.case_id == case_id
        assert len(assessment.criterion_results) == len(get_procurement_criteria())
