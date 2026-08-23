"""Parent LangGraph workflow for emergency procurement review."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph

from decision.emergency_criteria import get_procurement_criterion
from graph.assessment_helpers import (
    case_from_messages,
    create_chat_model,
)
from graph.audit_readiness_subagent import (
    build_audit_readiness_subgraph,
    create_audit_readiness_subagent_node,
)
from graph.emergency_verification_subagent import (
    build_emergency_verification_subgraph,
    create_emergency_verification_subagent_node,
)
from graph.shared import (
    AUDIT_READINESS_STAGE,
    EMERGENCY_VERIFICATION_STAGE,
    MAX_RESEARCH_ROUNDS,
)
from models.assessment import (
    AuditReadinessAssessment,
    CriterionResult,
    EmergencyProcurementAssessment,
    EmergencyVerification,
    EvidenceReference,
    FinalRecommendation,
)
from rag.tool_call_demo import AVAILABLE_TOOLS


class ProcurementGraphState(MessagesState):
    """Shared messages and state for the two-stage assessment workflow."""

    emergency_verification: EmergencyVerification | None
    audit_readiness: AuditReadinessAssessment | None
    assessment: EmergencyProcurementAssessment | None
    assessment_stage: str
    unresolved_criteria: list[CriterionResult]
    research_rounds: int
    max_research_rounds: int
    gap_research_active: bool
    gap_research_tools_used: bool


def route_after_emergency_verification(
    state: ProcurementGraphState,
) -> str:
    """Route the child graph's validated gate result in the parent graph."""

    verification = state.get("emergency_verification")
    if verification is not None and verification.emergency_is_verified is True:
        return AUDIT_READINESS_STAGE
    return "finalize"


def _append_list(
    lines: list[str],
    heading: str,
    values: Sequence[str],
) -> None:
    """Append a labeled list without interpreting its values."""

    if not values:
        return
    lines.append(f"{heading}:")
    lines.extend(f"- {value}" for value in values)


def _format_evidence(reference: EvidenceReference) -> str:
    """Render one existing evidence reference without changing its meaning."""

    source = f"source: {reference.source_id}"
    if reference.source_location:
        source += f", location: {reference.source_location}"
    rendered = f"{reference.description} ({source})"
    if reference.quote_or_fact:
        rendered += f" — {reference.quote_or_fact}"
    return rendered


def _render_criterion_result(result: CriterionResult) -> list[str]:
    """Render every structured field relevant to one criterion result."""

    try:
        criterion_name = get_procurement_criterion(result.criterion_id).name
    except KeyError:
        criterion_name = result.criterion_id.replace("_", " ").title()

    lines = [
        f"### {criterion_name} ({result.criterion_id})",
        f"Status: {result.status.value}",
        f"Rationale: {result.rationale}",
        f"Confidence: {result.confidence:.0%}",
    ]
    _append_list(
        lines,
        "Supporting evidence",
        [_format_evidence(item) for item in result.supporting_evidence],
    )
    _append_list(
        lines,
        "Conflicting evidence",
        [_format_evidence(item) for item in result.conflicting_evidence],
    )
    _append_list(lines, "Missing evidence", result.missing_evidence)
    _append_list(lines, "Follow-up questions", result.follow_up_questions)
    lines.append(
        "Requires human review: "
        f"{'yes' if result.requires_human_review else 'no'}"
    )
    if result.human_review_reason:
        lines.append(f"Human-review reason: {result.human_review_reason}")
    return lines


def _case_heading(state: ProcurementGraphState, case_id: str) -> list[str]:
    """Use case state only to add identifying presentation context."""

    case = case_from_messages(state["messages"])
    if case is None or case.case_id != case_id:
        return [f"Case: {case_id}"]
    return [
        f"Case: {case.case_id} — {case.title}",
        f"Agency: {case.jurisdiction.agency}",
        f"Department: {case.department}",
    ]


def _render_emergency_verification(
    state: ProcurementGraphState,
    verification: EmergencyVerification,
) -> str:
    """Render the verified, rejected, or indeterminate stage result."""

    if verification.emergency_is_verified is True:
        conclusion = (
            "The emergency procurement exception is justified based on the "
            "validated emergency-verification assessment."
        )
    elif verification.emergency_is_verified is False:
        conclusion = (
            "The emergency procurement exception is not justified based on "
            "the validated emergency-verification assessment."
        )
    else:
        conclusion = (
            "The available evidence is insufficient to determine whether the "
            "emergency procurement exception is justified."
        )

    lines = [
        *_case_heading(state, verification.case_id),
        "",
        "## Emergency verification",
        conclusion,
        f"Rationale: {verification.rationale}",
        f"Confidence: {verification.confidence:.0%}",
        "",
        "## Criterion results",
    ]
    for result in verification.criterion_results:
        lines.extend(["", *_render_criterion_result(result)])
    _append_list(lines, "Sources used", verification.source_ids_used)
    return "\n".join(lines)


_AUDIT_CONCLUSIONS = {
    FinalRecommendation.SUFFICIENTLY_SUPPORTED: (
        "The proposed emergency procurement file is audit-ready based on the "
        "validated audit-readiness assessment."
    ),
    FinalRecommendation.ADDITIONAL_EVIDENCE_REQUIRED: (
        "The proposed emergency procurement file is not yet audit-ready; the "
        "validated assessment requires additional evidence."
    ),
    FinalRecommendation.NOT_SUFFICIENTLY_SUPPORTED: (
        "The proposed emergency procurement file is not audit-ready based on "
        "the validated audit-readiness assessment."
    ),
    FinalRecommendation.HUMAN_REVIEW_REQUIRED: (
        "The proposed emergency procurement file requires human review before "
        "audit readiness can be determined."
    ),
}


def _render_audit_readiness(
    state: ProcurementGraphState,
    audit: AuditReadinessAssessment,
) -> str:
    """Render the audit result and its existing outstanding checklist."""

    lines = [
        *_case_heading(state, audit.case_id),
        "",
        "## Audit readiness",
        _AUDIT_CONCLUSIONS[audit.recommendation],
        f"Recommendation: {audit.recommendation.value}",
        f"Classification: {audit.classification}",
        f"Executive summary: {audit.executive_summary}",
        f"Overall confidence: {audit.overall_confidence:.0%}",
        "",
        "## Criterion results",
    ]
    for result in audit.criterion_results:
        lines.extend(["", *_render_criterion_result(result)])

    if audit.audit_risks:
        lines.extend(["", "## Audit risks"])
        for risk in audit.audit_risks:
            lines.extend(
                [
                    f"### {risk.title} ({risk.risk_id})",
                    f"Severity: {risk.severity}",
                    f"Description: {risk.description}",
                    "Related criteria: "
                    + (", ".join(risk.related_criterion_ids) or "none listed"),
                    f"Recommended action: {risk.recommended_action}",
                ]
            )

    lines.extend(["", "## Outstanding checklist"])
    _append_list(lines, "Missing documents", audit.missing_documents)
    _append_list(lines, "Required approvals", audit.required_approvals)
    _append_list(lines, "Next steps", audit.next_steps)
    lines.append(
        "Requires human review: "
        f"{'yes' if audit.requires_human_review else 'no'}"
    )
    if audit.human_review_reason:
        lines.append(f"Human-review reason: {audit.human_review_reason}")
    _append_list(lines, "Sources used", audit.source_ids_used)
    return "\n".join(lines)


def finalize_assessment(
    state: ProcurementGraphState,
) -> dict[str, list[AIMessage]]:
    """Render the active structured stage result without reassessing it."""

    stage = state.get("assessment_stage", EMERGENCY_VERIFICATION_STAGE)
    if stage == EMERGENCY_VERIFICATION_STAGE:
        verification = state.get("emergency_verification")
        if verification is None:
            content = (
                "No structured emergency-verification result is available. "
                "The emergency procurement determination could not be "
                "completed."
            )
        else:
            content = _render_emergency_verification(state, verification)
    elif stage == AUDIT_READINESS_STAGE:
        audit = state.get("audit_readiness")
        if audit is None:
            content = (
                "No structured audit-readiness result is available. The "
                "audit-readiness determination could not be completed."
            )
        else:
            content = _render_audit_readiness(state, audit)
    else:
        raise RuntimeError(f"Unsupported assessment stage: {stage}")

    return {"messages": [AIMessage(content=content)]}


def build_graph(
    *,
    chat_model: Any | None = None,
    tools: Sequence[BaseTool] = AVAILABLE_TOOLS,
) -> Any:
    """Build the parent graph around the emergency-verification sub-agent."""

    model = chat_model or create_chat_model()
    model_with_tools = model.bind_tools(list(tools))
    verification_subgraph = build_emergency_verification_subgraph(
        chat_model=model,
        tools=tools,
        model_with_tools=model_with_tools,
    )

    run_emergency_verification_subagent = (
        create_emergency_verification_subagent_node(verification_subgraph)
    )
    audit_subgraph = build_audit_readiness_subgraph(
        chat_model=model,
        tools=tools,
        model_with_tools=model_with_tools,
    )
    run_audit_readiness_subagent = create_audit_readiness_subagent_node(
        audit_subgraph
    )

    builder = StateGraph(ProcurementGraphState)
    builder.add_node(
        "emergency_verification_subagent",
        run_emergency_verification_subagent,
    )
    builder.add_node(
        "audit_readiness_subagent",
        run_audit_readiness_subagent,
    )
    builder.add_node("finalize_assessment", finalize_assessment)
    builder.add_edge(START, "emergency_verification_subagent")
    builder.add_conditional_edges(
        "emergency_verification_subagent",
        route_after_emergency_verification,
        {
            AUDIT_READINESS_STAGE: "audit_readiness_subagent",
            "finalize": "finalize_assessment",
        },
    )
    builder.add_edge("audit_readiness_subagent", "finalize_assessment")
    builder.add_edge("finalize_assessment", END)
    return builder.compile()


def run_graph(
    question: str,
    *,
    chat_model: Any | None = None,
    tools: Sequence[BaseTool] = AVAILABLE_TOOLS,
) -> AIMessage:
    """Run the graph and return its deterministic structured finalization."""

    if not question.strip():
        raise ValueError("question must not be blank")

    result = build_graph(chat_model=chat_model, tools=tools).invoke(
        {
            "messages": [HumanMessage(content=question)],
            "emergency_verification": None,
            "audit_readiness": None,
            "assessment": None,
            "assessment_stage": EMERGENCY_VERIFICATION_STAGE,
            "research_rounds": 0,
            "max_research_rounds": MAX_RESEARCH_ROUNDS,
            "gap_research_active": False,
            "gap_research_tools_used": False,
        }
    )
    final_message = result["messages"][-1]
    if not isinstance(final_message, AIMessage):
        raise RuntimeError("graph ended without a model response")
    return final_message
