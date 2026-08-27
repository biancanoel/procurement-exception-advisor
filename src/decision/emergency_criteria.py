"""Emergency procurement decision criteria."""

from models.criteria import (
    EmergencyCriterion,
    EvidenceType,
)
PROCUREMENT_CRITERIA: tuple[EmergencyCriterion, ...] = (
    EmergencyCriterion(
        criterion_id="unexpected_event",
        name="Unexpected or urgent event",
        description=(
            "Determine whether the emergency resulted from an unforeseen event "
            "rather than poor planning, neglect, failure to maintain equipment, "
            "contract expiration, inventory mismanagement, or administrative delay."
        ),
        questions_to_answer=[
            "What happened?",
            "When did the event occur or become known?",
            "Could the department reasonably have anticipated the need?",
            "Did delayed planning, contract management, or inventory management contribute to the urgency?",
        ],
        expected_evidence=[
            "Incident timeline",
            "Incident report or operational record",
            "Explanation of when the need became known",
        ],
        preferred_evidence_types=[
            EvidenceType.INCIDENT_REPORT,
            EvidenceType.TIMELINE,
            EvidenceType.EMAIL,
            EvidenceType.TECHNICAL_ASSESSMENT,
        ],
        risk_if_missing=(
            "Routine urgency or poor planning may be incorrectly treated "
            "as a qualifying emergency."
        ),
        human_review_triggers=[
            "The urgency appears to result primarily from poor planning.",
            "The department knew about the need well before requesting emergency authority.",
            "The evidence contains conflicting dates.",
        ],
    ),
    EmergencyCriterion(
        criterion_id="immediate_harm",
        name="Immediate harm if delayed",
        description=(
            "Determine whether delaying the purchase would create a "
            "specific and meaningful threat to health, safety, property, "
            "essential operations, the environment, or public welfare."
        ),
        questions_to_answer=[
            "Is there an imminent threat to public health, safety, or welfare?",
            "Will delay cause serious or secondary damage to public property?",
            "Will delay materially interrupt an essential public-facing service?",
        ],
        expected_evidence=[
            "Description of the threatened harm",
            "Timeframe in which harm may occur",
            "Evidence of temporary controls or why they are inadequate",
        ],
        preferred_evidence_types=[
            EvidenceType.TECHNICAL_ASSESSMENT,
            EvidenceType.INCIDENT_REPORT,
            EvidenceType.PHOTOGRAPH,
            EvidenceType.TIMELINE,
        ],
        risk_if_missing=(
            "The agent cannot distinguish inconvenience or delay from "
            "a genuine emergency."
        ),
        human_review_triggers=[
            "The claimed harm involves public health or safety.",
            "The risk is disputed by available evidence.",
            "Temporary measures may make competition feasible.",
        ],
    ),
    EmergencyCriterion(
        criterion_id="competition_impracticable",
        name="Competition is impracticable",
        description=(
            "Determine whether normal or abbreviated competition cannot "
            "be completed without causing unacceptable delay or harm. Assume that normal competition would like at least 1 month to complete."
""        ),
        questions_to_answer=[
            "Why would normal competition create an unacceptable delay?",
            "Could informal quotes or an abbreviated competitive process be used?",
            "Were alternative vendors contacted?",
            "What were the availability and response times of other vendors?",
            "Is there an existing contract or cooperative contract that could meet the need?",
        ],
        expected_evidence=[
            "Explanation of why competition is impracticable",
            "Vendor contact attempts",
            "Vendor availability information",
            "Timeline showing the effect of delay",
            "Review of available contracts or alternatives",
            "Record of existing-contract search",
            "Record of cooperative-contract search",
        ],
        preferred_evidence_types=[
            EvidenceType.VENDOR_AVAILABILITY,
            EvidenceType.EMAIL,
            EvidenceType.QUOTE,
            EvidenceType.MARKET_RESEARCH,
            EvidenceType.TIMELINE,
        ],
        risk_if_missing=(
            "The emergency exception may be used when limited competition "
            "was reasonably available."
        ),
        human_review_triggers=[
            "No vendors other than the proposed vendor were considered.",
            "Temporary controls may provide time for competition.",
            "The department states that competition is inconvenient rather than impracticable.",
        ],
    ),
    EmergencyCriterion(
        criterion_id="appropriate_response_scope",
        name="Response is appropriate and limited in scope",
        description=(
            "Determine whether the proposed purchase directly addresses the "
            "emergency and whether its scope, quantity, value, and duration "
            "are reasonably limited to what is necessary to prevent, reduce, "
            "stabilize, or resolve the identified harm."
        ),
        questions_to_answer=[
            "How does the proposed purchase address the emergency?",
            "Does the request include upgrades, future phases, or unrelated work?",
            "Could non-emergency work be separated and competitively procured?",
            "Is the requested contract term limited to the emergency period?",
            "Is the requested quantity limited to the amount needed to bridge the emergency?",
        ],
        expected_evidence=[
            "Technical explanation connecting the response to the emergency",
            "Itemized scope",
            "Itemized pricing",
            "Emergency stabilization plan",
            "Alternative solutions considered",
        ],
        preferred_evidence_types=[
            EvidenceType.QUOTE,
            EvidenceType.TECHNICAL_ASSESSMENT,
            EvidenceType.CONTRACT,
            EvidenceType.MARKET_RESEARCH,
        ],
        risk_if_missing=(
            "The agency may authorize a purchase that is unrelated, "
            "unnecessary, disproportionate, or broader or longer than the "
            "emergency requires."
        ),
        human_review_triggers=[
            "Technical evidence does not support the proposed solution.",
            "A less extensive alternative appears available.",
            "The proposal includes capital improvements unrelated to stabilization.",
            "The requested term extends beyond the expected emergency period.",
            "The vendor has not itemized emergency and non-emergency work.",
        ],
    ),
    EmergencyCriterion(
        criterion_id="vendor_selection",
        name="Vendor selection is supported",
        description=(
            "Determine whether the proposed vendor was selected for a "
            "documented, reasonable reason rather than preference or convenience."
        ),
        questions_to_answer=[
            "Were other qualified vendors considered?",
        ],
        expected_evidence=[
            "Vendor availability confirmation",
            "Records of alternative vendor contacts",
        ],
        preferred_evidence_types=[
            EvidenceType.VENDOR_AVAILABILITY,
            EvidenceType.LICENSE_OR_REGISTRATION,
            EvidenceType.EMAIL,
            EvidenceType.MARKET_RESEARCH,
        ],
        risk_if_missing=(
            "The procurement file may not demonstrate why the selected "
            "vendor was reasonable under the circumstances."
        ),
        human_review_triggers=[
            "The only reason provided is that the department prefers the vendor.",
            "The vendor lacks required qualifications or registrations.",
            "The proposed vendor has a potential conflict of interest.",
        ],
    ),
    EmergencyCriterion(
        criterion_id="price_reasonableness",
        name="Price reasonableness",
        description=(
            "Determine whether the agency has taken reasonable steps to "
            "establish that the proposed price is fair under the circumstances."
        ),
        questions_to_answer=[
            "What is the proposed total price?",
            "Were any competing quotes obtained?",
            "Is the price consistent with prior purchases, contracts, published rates, or an independent estimate?",
            "Are labor, materials, equipment, and markup itemized?",
            "Does the urgency create unusual costs that should be documented?",
        ],
        expected_evidence=[
            "Vendor quote",
            "Comparable quote or recent purchase",
            "Independent estimate",
            "Existing contract pricing",
            "Written price-reasonableness determination",
        ],
        preferred_evidence_types=[
            EvidenceType.QUOTE,
            EvidenceType.PRICE_COMPARISON,
            EvidenceType.CONTRACT,
            EvidenceType.MARKET_RESEARCH,
        ],
        risk_if_missing=(
            "The agency may be unable to demonstrate that it protected "
            "public funds despite reduced competition."
        ),
        human_review_triggers=[
            "The proposed price materially exceeds available comparisons.",
            "No price-support method is documented.",
            "The quote contains large unitemized allowances or markups.",
        ],
    ),
    EmergencyCriterion(
        criterion_id="approval_authority",
        name="Required authority and approvals",
        description=(
            "Determine which officials or governing bodies must authorize, "
            "ratify, report, or periodically review the emergency action."
        ),
        questions_to_answer=[
            "Who has authority to approve the requested amount and category?",
            "Is prior governing-body approval required?",
            "Is post-award reporting or ratification required?",
            "Must the agency periodically review whether the emergency still exists?",
            "Has work begun before the necessary approval was obtained?",
        ],
        expected_evidence=[
            "Delegation or approval matrix",
            "Signed approval record",
            "Staff report or governing-body action",
            "Required follow-up review schedule",
        ],
        preferred_evidence_types=[
            EvidenceType.POLICY,
            EvidenceType.STATUTE,
            EvidenceType.APPROVAL_RECORD,
        ],
        risk_if_missing=(
            "The procurement may be unauthorized even if the underlying "
            "emergency is genuine."
        ),
        allows_partial_support=False,
        human_review_triggers=[
            "Local policy and state law appear inconsistent.",
            "The requested amount exceeds delegated authority.",
            "Work began without documented authorization.",
            "Governing-body reporting or review may be required.",
        ],
    ),
    EmergencyCriterion(
        criterion_id="post_facto_formalization",
        name="Post-event formalization",
        description=(
            "Determine whether the agency completed the required justification, "
            "ratification, reporting, and continuing review after emergency "
            "action was taken."
        ),
        questions_to_answer=[
            "Was a contemporaneous written justification prepared?",
            "Does it explain the facts, threat, alternatives, vendor selection, and price?",
            "Was the justification signed by the proper official?",
            "Must the purchase be reported to the City Manager or governing body?",
            "Is formal ratification required?",
            "Must the agency periodically confirm that the emergency still exists?",
            "Was emergency authority ended once competition became feasible?",
        ],
        expected_evidence=[
            "Signed contemporaneous justification memo",
            "Approval or ratification record",
            "City Manager or governing-body report",
            "Meeting agenda or staff report",
            "Continued-emergency review records",
            "Record closing the emergency action",
        ],
        preferred_evidence_types=[
            EvidenceType.APPROVAL_RECORD,
            EvidenceType.TIMELINE,
            EvidenceType.POLICY,
            EvidenceType.STATUTE,
        ],
        risk_if_missing=(
            "A valid emergency purchase may still fail an audit because required "
            "reporting, ratification, or contemporaneous documentation was omitted."
        ),
        allows_partial_support=False,
        human_review_triggers=[
            "The justification was prepared long after the purchase.",
            "Required ratification or reporting has not occurred.",
            "The emergency contract continued after competition became feasible.",
        ],
      )
)


EMERGENCY_VERIFICATION_CRITERION_IDS = (
    "unexpected_event",
    "immediate_harm",
    "competition_impracticable",
)

AUDIT_READINESS_CRITERION_IDS = (
    "appropriate_response_scope",
    "vendor_selection",
    "price_reasonableness",
    "approval_authority",
    "post_facto_formalization",
)

_CRITERIA_BY_ID = {
    criterion.criterion_id: criterion
    for criterion in PROCUREMENT_CRITERIA
}

EMERGENCY_CRITERIA = tuple(
    _CRITERIA_BY_ID[criterion_id]
    for criterion_id in EMERGENCY_VERIFICATION_CRITERION_IDS
)

AUDIT_READINESS_CRITERIA = tuple(
    _CRITERIA_BY_ID[criterion_id]
    for criterion_id in AUDIT_READINESS_CRITERION_IDS
)


def get_emergency_criteria() -> tuple[EmergencyCriterion, ...]:
    """Return the three criteria that verify an emergency exists."""

    return EMERGENCY_CRITERIA


def get_audit_readiness_criteria() -> tuple[EmergencyCriterion, ...]:
    """Return the five criteria used to evaluate audit readiness."""

    return AUDIT_READINESS_CRITERIA


def get_procurement_criteria() -> tuple[EmergencyCriterion, ...]:
    """Return all criteria in two-stage workflow order."""

    return EMERGENCY_CRITERIA + AUDIT_READINESS_CRITERIA


def get_emergency_criterion(
    criterion_id: str,
) -> EmergencyCriterion:
    """Return one emergency-verification criterion by its identifier."""

    for criterion in EMERGENCY_CRITERIA:
        if criterion.criterion_id == criterion_id:
            return criterion

    raise KeyError(
        f"Unknown emergency criterion: {criterion_id}"
    )


def get_audit_readiness_criterion(
    criterion_id: str,
) -> EmergencyCriterion:
    """Return one audit-readiness criterion by its identifier."""

    for criterion in AUDIT_READINESS_CRITERIA:
        if criterion.criterion_id == criterion_id:
            return criterion

    raise KeyError(
        f"Unknown audit-readiness criterion: {criterion_id}"
    )


def get_procurement_criterion(
    criterion_id: str,
) -> EmergencyCriterion:
    """Return any configured procurement criterion by its identifier."""

    try:
        return _CRITERIA_BY_ID[criterion_id]
    except KeyError as error:
        raise KeyError(
            f"Unknown procurement criterion: {criterion_id}"
        ) from error


def get_required_emergency_criteria(
) -> tuple[EmergencyCriterion, ...]:
    """Return criteria required for a recommendation."""

    return tuple(
        criterion
        for criterion in EMERGENCY_CRITERIA
        if criterion.required_for_recommendation
    )


def get_emergency_criteria_count() -> int:
    """Return the number of emergency-verification criteria."""

    return len(EMERGENCY_CRITERIA)


def get_audit_readiness_criteria_count() -> int:
    """Return the number of audit-readiness criteria."""

    return len(AUDIT_READINESS_CRITERIA)


def get_procurement_criteria_count() -> int:
    """Return the total number of criteria across both stages."""

    return len(PROCUREMENT_CRITERIA)
