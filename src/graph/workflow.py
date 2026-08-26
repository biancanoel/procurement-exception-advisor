"""Parent LangGraph workflow for emergency procurement review."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph

from decision.emergency_criteria import get_procurement_criterion
from graph.assessment_helpers import (
    append_html_list,
    case_from_state,
    create_chat_model,
    format_evidence,
)
from graph.audit_readiness_subagent import (
    build_audit_readiness_subgraph,
    create_audit_readiness_subagent_node,
)
from graph.emergency_verification_subagent import (
    build_emergency_verification_subgraph,
    create_emergency_verification_subagent_node,
    emergency_verification_tools,
)
from graph.procurement_context_subagent import (
    PROCUREMENT_CONTEXT_STAGE,
    build_procurement_context_subgraph,
    create_procurement_context_subagent_node,
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
    FinalRecommendation,
    ProcurementContext,
)
from models.cases import EmergencyCaseInput
from models.criteria import CriterionStatus
from rag.tool_call_demo import AVAILABLE_TOOLS


class ProcurementGraphState(MessagesState):
    """Shared messages and state for the staged procurement workflow."""

    case_input: EmergencyCaseInput | None
    emergency_verification: EmergencyVerification | None
    procurement_context: ProcurementContext | None
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
        return PROCUREMENT_CONTEXT_STAGE
    return "finalize"


def _render_criterion_result(result: CriterionResult) -> list[str]:
    """Render every structured field relevant to one criterion result."""

    try:
        criterion_name = get_procurement_criterion(result.criterion_id).name
    except KeyError:
        criterion_name = result.criterion_id.replace("_", " ").title()

    status_label = result.status.value
    if result.status == CriterionStatus.NOT_EVALUATED:
        status_label = "insufficient evidence — criterion not passed"

    lines = [
        f"### {criterion_name} ({result.criterion_id})",
        f"Status: {status_label}",
        f"Rationale: {result.rationale}",
        f"Confidence: {result.confidence:.0%}",
    ]
    append_html_list(
        lines,
        "Supporting evidence",
        [format_evidence(item) for item in result.supporting_evidence],
    )
    append_html_list(
        lines,
        "Conflicting evidence",
        [format_evidence(item) for item in result.conflicting_evidence],
    )
    append_html_list(lines, "Missing evidence", result.missing_evidence)
    append_html_list(lines, "Follow-up questions", result.follow_up_questions)
    lines.append(
        "Requires human review: "
        f"{'yes' if result.requires_human_review else 'no'}"
    )
    if result.human_review_reason:
        lines.append(f"Human-review reason: {result.human_review_reason}")
    return lines


def _case_heading(state: ProcurementGraphState, case_id: str) -> list[str]:
    """Use case state only to add identifying presentation context."""

    case = case_from_state(state)
    if case is None or case.case_id != case_id:
        return [f"Case: {case_id}"]
    heading = [
        (
            f"Case: {case.case_id} — {case.title}"
            if case.title
            else f"Case: {case.case_id}"
        )
    ]
    if case.jurisdiction is not None and case.jurisdiction.agency:
        heading.append(f"Agency: {case.jurisdiction.agency}")
    if case.department:
        heading.append(f"Department: {case.department}")
    return heading


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
    append_html_list(lines, "Missing documents", audit.missing_documents)
    append_html_list(lines, "Required approvals", audit.required_approvals)
    append_html_list(lines, "Next steps", audit.next_steps)
    lines.append(
        "Requires human review: "
        f"{'yes' if audit.requires_human_review else 'no'}"
    )
    if audit.human_review_reason:
        lines.append(f"Human-review reason: {audit.human_review_reason}")
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
    """Build the parent graph around the three staged child graphs."""

    model = chat_model or create_chat_model()
    # Removing get_case_facts from the emergency verification subagent tools since we alreaddt have case facts from gradio. Keeping get_case_facts so we can still use test cases
    verification_tools = emergency_verification_tools(tools)
    verification_model_with_tools = model.bind_tools(verification_tools)
    verification_model_for_supplied_case = model.bind_tools(
        [
            tool
            for tool in verification_tools
            if tool.name != "get_case_facts"
        ]
    )
    verification_subgraph = build_emergency_verification_subgraph(
        chat_model=model,
        tools=verification_tools,
        model_with_tools=verification_model_with_tools,
        model_for_supplied_case=verification_model_for_supplied_case,
    )

    run_emergency_verification_subagent = (
        create_emergency_verification_subagent_node(verification_subgraph)
    )
    context_subgraph = build_procurement_context_subgraph(
        chat_model=model,
        tools=tools,
    )
    run_procurement_context_subagent = (
        create_procurement_context_subagent_node(context_subgraph)
    )
    # Removing get_case_facts from the audit readiness subagent tools since we alreaddt have case facts from gradio. Keeping get_case_facts so we can still use test cases
    audit_model_with_tools = model.bind_tools(list(tools))
    audit_model_for_supplied_case = model.bind_tools(
        [tool for tool in tools if tool.name != "get_case_facts"]
    )
    audit_subgraph = build_audit_readiness_subgraph(
        chat_model=model,
        tools=tools,
        model_with_tools=audit_model_with_tools,
        model_for_supplied_case=audit_model_for_supplied_case,
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
        "procurement_context_subagent",
        run_procurement_context_subagent,
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
            PROCUREMENT_CONTEXT_STAGE: "procurement_context_subagent",
            "finalize": "finalize_assessment",
        },
    )
    builder.add_edge(
        "procurement_context_subagent",
        "audit_readiness_subagent",
    )
    builder.add_edge("audit_readiness_subagent", "finalize_assessment")
    builder.add_edge("finalize_assessment", END)
    return builder.compile()


def _initial_graph_state(
    question: str,
    case_input: EmergencyCaseInput | None,
) -> dict[str, Any]:
    """Create the shared initial state used by synchronous and streamed runs."""

    return {
        "messages": [HumanMessage(content=question)],
        "case_input": case_input,
        "emergency_verification": None,
        "procurement_context": None,
        "audit_readiness": None,
        "assessment": None,
        "assessment_stage": EMERGENCY_VERIFICATION_STAGE,
        "research_rounds": 0,
        "max_research_rounds": MAX_RESEARCH_ROUNDS,
        "gap_research_active": False,
        "gap_research_tools_used": False,
    }


def stream_graph(
    question: str,
    *,
    case_input: EmergencyCaseInput | None = None,
    chat_model: Any | None = None,
    tools: Sequence[BaseTool] = AVAILABLE_TOOLS,
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Yield parent-node updates so interfaces can display stage progress."""

    if not question.strip():
        raise ValueError("question must not be blank")

    graph = build_graph(chat_model=chat_model, tools=tools)
    for update in graph.stream(
        _initial_graph_state(question, case_input),
        stream_mode="updates",
    ):
        for node_name, node_update in update.items():
            yield node_name, node_update


def run_graph(
    question: str,
    *,
    case_input: EmergencyCaseInput | None = None,
    chat_model: Any | None = None,
    tools: Sequence[BaseTool] = AVAILABLE_TOOLS,
) -> AIMessage:
    """Run the graph and return its deterministic structured finalization."""

    if not question.strip():
        raise ValueError("question must not be blank")

    result = build_graph(chat_model=chat_model, tools=tools).invoke(
        _initial_graph_state(question, case_input)
    )
    final_message = result["messages"][-1]
    if not isinstance(final_message, AIMessage):
        raise RuntimeError("graph ended without a model response")
    return final_message
