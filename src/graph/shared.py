"""Shared evidence-gap nodes for staged procurement assessments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from models.assessment import CriterionResult
from models.criteria import CriterionStatus


MAX_RESEARCH_ROUNDS = 3

EMERGENCY_VERIFICATION_STAGE = "emergency_verification"
AUDIT_READINESS_STAGE = "audit_readiness"

_UNRESOLVED_STATUSES = {
    CriterionStatus.NOT_EVALUATED,
    CriterionStatus.PARTIALLY_SUPPORTED,
    CriterionStatus.HUMAN_REVIEW_REQUIRED,
}


def _is_unresolved(result: CriterionResult) -> bool:
    """Use status as the primary signal for unresolved assessment work."""

    return result.status in _UNRESOLVED_STATUSES


def check_evidence_gaps(
    state: Mapping[str, Any],
) -> dict[str, list[CriterionResult] | bool | int]:
    """Collect unresolved results only from the current assessment stage."""

    stage = state.get("assessment_stage", EMERGENCY_VERIFICATION_STAGE)
    if stage == EMERGENCY_VERIFICATION_STAGE:
        verification = state.get("emergency_verification")
        results = [] if verification is None else verification.criterion_results
    elif stage == AUDIT_READINESS_STAGE:
        audit = state.get("audit_readiness")
        results = [] if audit is None else audit.criterion_results
    else:
        raise RuntimeError(f"Unsupported assessment stage: {stage}")

    return {
        "unresolved_criteria": [
            result for result in results if _is_unresolved(result)
        ],
        "research_rounds": state.get("research_rounds", 0),
        "max_research_rounds": state.get(
            "max_research_rounds", MAX_RESEARCH_ROUNDS
        ),
        "gap_research_active": False,
        "gap_research_tools_used": False,
    }


def prepare_gap_research(
    state: Mapping[str, Any],
) -> dict[str, list[BaseMessage] | bool | int]:
    """Send the current stage's unresolved batch into one research round."""

    next_round = state.get("research_rounds", 0) + 1
    stage = state.get("assessment_stage", EMERGENCY_VERIFICATION_STAGE)
    unresolved_context = [
        result.model_dump(mode="json")
        for result in state["unresolved_criteria"]
    ]
    message = HumanMessage(
        content=(
            f"These are the complete unresolved {stage} criterion results "
            f"(additional research round {next_round}):\n\n"
            f"{json.dumps(unresolved_context, indent=2)}\n\n"
            "Decide whether any gaps can be addressed with your available "
            "tools. If so, make one or multiple appropriate tool calls. Do "
            "not invent evidence and do not force a tool call. If the tools "
            "cannot resolve the remaining gaps, return a normal response that "
            "preserves the missing evidence and follow-up needs."
        )
    )
    return {
        "messages": [message],
        "research_rounds": next_round,
        "gap_research_active": True,
        "gap_research_tools_used": False,
    }
