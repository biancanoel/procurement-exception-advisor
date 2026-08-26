"""Offline tests for the two-stage LangGraph assessment workflow."""

from typing import Any, get_type_hints

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import ValidationError

from data.case_loader import load_case
from decision.emergency_criteria import (
    AUDIT_READINESS_CRITERIA,
    EMERGENCY_CRITERIA,
)
import graph.audit_readiness_subagent as audit_subagent_module
import graph.emergency_verification_subagent as emergency_subagent_module
from graph.assessment_helpers import (
    append_html_list,
    create_model_node,
    route_model_response,
)
from graph.audit_readiness_subagent import (
    AuditReadinessNodeUpdate,
    audit_readiness,
    build_audit_readiness_subgraph,
    route_audit_readiness_gaps,
)
from graph.emergency_verification_subagent import (
    EmergencyVerificationNodeUpdate,
    build_emergency_verification_subgraph,
    emergency_verification,
    emergency_verification_tools,
    route_emergency_verification_entry,
    route_emergency_verification_gaps,
)
from graph.shared import (
    AUDIT_READINESS_STAGE,
    EMERGENCY_VERIFICATION_STAGE,
    MAX_RESEARCH_ROUNDS,
    check_evidence_gaps,
    prepare_gap_research,
)
from graph.workflow import (
    build_graph,
    finalize_assessment,
    route_after_emergency_verification,
    run_graph,
    stream_graph,
)
from models.assessment import (
    AuditRisk,
    AuditReadinessAssessment,
    CriterionResult,
    EmergencyProcurementAssessment,
    EmergencyVerification,
    EvidenceReference,
    FinalRecommendation,
)
from models.cases import EmergencyCaseInput
from models.criteria import CriterionStatus
from rag.tools import (
    get_case_facts,
    search_government_awards,
    search_procurement_rules,
)


@tool
def example_lookup(query: str) -> dict[str, str]:
    """Look up an example value."""

    return {"result": f"Found {query}"}


class FakeBoundModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = iter(responses)
        self.inputs: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.inputs.append(messages)
        return next(self.responses)


class FakeStructuredModel:
    def __init__(self, owner: "FakeChatModel") -> None:
        self.owner = owner

    def invoke(self, messages: list[Any]) -> Any:
        self.owner.structured_inputs.append(messages)
        response = next(self.owner.structured_responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeChatModel:
    def __init__(
        self,
        responses: list[AIMessage],
        structured_responses: list[
            EmergencyVerification | AuditReadinessAssessment | Exception
        ] | None = None,
    ) -> None:
        self.bound_tools: list[Any] = []
        self.bound_tool_sets: list[list[Any]] = []
        self.bound_model = FakeBoundModel(responses)
        self.structured_responses = iter(structured_responses or [])
        self.structured_schemas: list[type[Any]] = []
        self.structured_methods: list[str] = []
        self.structured_inputs: list[list[Any]] = []

    def bind_tools(self, tools: list[Any]) -> FakeBoundModel:
        self.bound_tools = tools
        self.bound_tool_sets.append(list(tools))
        return self.bound_model

    def with_structured_output(
        self,
        schema: type[Any],
        *,
        method: str,
    ) -> FakeStructuredModel:
        self.structured_schemas.append(schema)
        self.structured_methods.append(method)
        return FakeStructuredModel(self)


def tool_request(
    *,
    call_id: str = "call-001",
    query: str = "emergency pumps",
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "example_lookup",
                "args": {"query": query},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def criterion_result(
    criterion_id: str,
    *,
    status: CriterionStatus = CriterionStatus.NOT_EVALUATED,
    missing_evidence: list[str] | None = None,
    follow_up_questions: list[str] | None = None,
) -> CriterionResult:
    supporting_evidence = []
    if status in {CriterionStatus.NOT_SUPPORTED, CriterionStatus.CONTRADICTED}:
        supporting_evidence = [
            EvidenceReference(
                source_id="CASE-DOC-01",
                source_type="case_document",
                description="Affirmative evidence that the criterion fails.",
            )
        ]
    return CriterionResult(
        criterion_id=criterion_id,
        status=status,
        rationale=f"Assessment rationale for {criterion_id}.",
        supporting_evidence=supporting_evidence,
        missing_evidence=missing_evidence or [],
        follow_up_questions=follow_up_questions or [],
        confidence=0.0 if status == CriterionStatus.NOT_EVALUATED else 0.9,
    )


def verification(
    determination: bool | None,
    *,
    unresolved_id: str = "unexpected_event",
) -> EmergencyVerification:
    results = [
        criterion_result(
            criterion.criterion_id,
            status=CriterionStatus.SUPPORTED,
        )
        for criterion in EMERGENCY_CRITERIA
    ]
    if determination is None:
        results = [
            criterion_result(
                result.criterion_id,
                status=(
                    CriterionStatus.NOT_EVALUATED
                    if result.criterion_id == unresolved_id
                    else CriterionStatus.SUPPORTED
                ),
                missing_evidence=(
                    ["Additional emergency evidence"]
                    if result.criterion_id == unresolved_id
                    else []
                ),
            )
            for result in results
        ]
    elif determination is False:
        results[-1] = criterion_result(
            "competition_impracticable",
            status=CriterionStatus.NOT_SUPPORTED,
        )
    return EmergencyVerification(
        case_id="EM-001",
        emergency_is_verified=determination,
        criterion_results=results,
        rationale="Emergency verification rationale.",
        confidence=0.85 if determination is not None else 0.4,
    )


def rejected_em003_verification() -> EmergencyVerification:
    """Build the poor-planning verification used in routing tests."""

    return EmergencyVerification(
        case_id="EM-003",
        emergency_is_verified=False,
        criterion_results=[
            CriterionResult(
                criterion_id="unexpected_event",
                status=CriterionStatus.NOT_SUPPORTED,
                rationale=(
                    "The contract expiration was foreseeable and advance "
                    "reminders were provided."
                ),
                supporting_evidence=[
                    EvidenceReference(
                        source_id="EM003-D02",
                        source_type="case_document",
                        description="Advance reminders were sent six months earlier.",
                    )
                ],
                confidence=0.95,
            ),
            CriterionResult(
                criterion_id="immediate_harm",
                status=CriterionStatus.SUPPORTED,
                rationale="A service interruption creates an operational concern.",
                confidence=0.85,
            ),
            CriterionResult(
                criterion_id="competition_impracticable",
                status=CriterionStatus.PARTIALLY_SUPPORTED,
                rationale=(
                    "Immediate continuity is supported, but alternatives have "
                    "not been fully documented."
                ),
                missing_evidence=["Documented review of available alternatives"],
                follow_up_questions=["Were alternate providers contacted?"],
                confidence=0.6,
            ),
        ],
        rationale=(
            "Service interruption presents a real operational concern, but "
            "the triggering circumstance was foreseeable and caused by "
            "missed procurement planning."
        ),
        confidence=0.82,
        source_ids_used=["EM003-D01", "EM003-D02"],
    )


def audit_assessment(
    *,
    unresolved_id: str | None = None,
    case_id: str = "EM-001",
) -> AuditReadinessAssessment:
    results = [
        criterion_result(
            criterion.criterion_id,
            status=(
                CriterionStatus.NOT_EVALUATED
                if criterion.criterion_id == unresolved_id
                else CriterionStatus.SUPPORTED
            ),
            missing_evidence=(
                ["Additional audit evidence"]
                if criterion.criterion_id == unresolved_id
                else []
            ),
        )
        for criterion in AUDIT_READINESS_CRITERIA
    ]
    return AuditReadinessAssessment(
        case_id=case_id,
        recommendation=(
            FinalRecommendation.ADDITIONAL_EVIDENCE_REQUIRED
            if unresolved_id
            else FinalRecommendation.SUFFICIENTLY_SUPPORTED
        ),
        executive_summary="Audit-readiness assessment.",
        classification="Emergency procurement",
        criterion_results=results,
        overall_confidence=0.5 if unresolved_id else 0.9,
    )


def state_with_case(case_id: str = "EM-001") -> dict[str, Any]:
    case = load_case(case_id)
    return {
        "messages": [
            HumanMessage(content=f"Evaluate {case_id}"),
            ToolMessage(
                content=case.model_dump_json(),
                name="get_case_facts",
                tool_call_id="case-call-001",
                artifact=case,
            ),
            ToolMessage(
                content='{"results": [{"section": "2.24.090"}]}',
                name="search_procurement_rules",
                tool_call_id="rules-call-001",
            ),
            AIMessage(content="Research complete."),
        ],
        "research_rounds": 0,
        "max_research_rounds": MAX_RESEARCH_ROUNDS,
        "gap_research_active": False,
        "gap_research_tools_used": False,
    }


def test_graph_executes_tool_and_loops_back_to_model() -> None:
    final = AIMessage(content="No case facts are available for assessment.")
    model = FakeChatModel([tool_request(), final])

    result = build_graph(chat_model=model, tools=[example_lookup]).invoke(
        {"messages": [HumanMessage(content="Find similar awards")]}
    )

    assert model.bound_tools == [example_lookup]
    assert result["messages"][-2] is final
    assert (
        "No structured emergency-verification result is available"
        in result["messages"][-1].content
    )
    assert len(model.bound_model.inputs) == 2
    observation = model.bound_model.inputs[1][-1]
    assert isinstance(observation, ToolMessage)
    assert observation.tool_call_id == "call-001"
    assert "Found emergency pumps" in str(observation.content)


def test_router_selects_tools_and_current_assessment_stage() -> None:
    assert route_model_response({"messages": [tool_request()]}) == "tools"
    assert route_model_response(
        {"messages": [AIMessage(content="Done")]}
    ) == EMERGENCY_VERIFICATION_STAGE
    assert route_model_response(
        {
            "messages": [AIMessage(content="Done")],
            "gap_research_active": True,
            "gap_research_tools_used": True,
            "assessment_stage": AUDIT_READINESS_STAGE,
        }
    ) == AUDIT_READINESS_STAGE


def test_model_node_uses_case_independent_tools_for_supplied_case() -> None:
    regular_model = FakeBoundModel([AIMessage(content="regular")])
    supplied_case_model = FakeBoundModel([AIMessage(content="supplied")])
    call_model = create_model_node(
        regular_model,
        model_for_supplied_case=supplied_case_model,
    )

    update = call_model(
        {
            "messages": [HumanMessage(content="Evaluate this case")],
            "case_input": EmergencyCaseInput(
                description="A beach tournament needs replacement nets."
            ),
        }
    )

    assert update["messages"][0].content == "supplied"
    assert regular_model.inputs == []
    assert len(supplied_case_model.inputs) == 1


def test_parent_starts_with_emergency_verification_subagent() -> None:
    graph = build_graph(
        chat_model=FakeChatModel([AIMessage(content="Research complete.")]),
        tools=[example_lookup],
    ).get_graph()
    start_targets = [
        edge.target for edge in graph.edges if edge.source == "__start__"
    ]

    assert start_targets == ["emergency_verification_subagent"]
    assert EMERGENCY_VERIFICATION_STAGE not in graph.nodes
    assert AUDIT_READINESS_STAGE not in graph.nodes
    assert "audit_readiness_subagent" in graph.nodes
    assert "model" not in graph.nodes
    assert "tools" not in graph.nodes
    assert "check_evidence_gaps" not in graph.nodes
    assert "finalize_assessment" in graph.nodes
    assert any(
        edge.source == "audit_readiness_subagent"
        and edge.target == "finalize_assessment"
        for edge in graph.edges
    )


def test_emergency_verification_is_a_compiled_subgraph() -> None:
    subgraph = build_emergency_verification_subgraph(
        chat_model=FakeChatModel([AIMessage(content="Research complete.")]),
        tools=[example_lookup],
    ).get_graph()

    assert {
        "model",
        "tools",
        EMERGENCY_VERIFICATION_STAGE,
        "check_evidence_gaps",
        "prepare_gap_research",
    } <= set(subgraph.nodes)


def test_supplied_case_enters_directly_at_emergency_verification() -> None:
    case = EmergencyCaseInput(description="A critical pump failed.")

    assert route_emergency_verification_entry(
        {"messages": [HumanMessage(content="Evaluate")], "case_input": case}
    ) == EMERGENCY_VERIFICATION_STAGE
    assert route_emergency_verification_entry(
        {"messages": [HumanMessage(content="Evaluate")], "case_input": None}
    ) == "model"


def test_emergency_verification_excludes_government_award_search() -> None:
    selected = emergency_verification_tools(
        [
            search_procurement_rules,
            get_case_facts,
            search_government_awards,
        ]
    )

    assert [tool.name for tool in selected] == [
        "search_procurement_rules",
        "get_case_facts",
    ]


def test_audit_readiness_is_a_compiled_subgraph() -> None:
    subgraph = build_audit_readiness_subgraph(
        chat_model=FakeChatModel([]),
        tools=[example_lookup],
    ).get_graph()

    assert {
        AUDIT_READINESS_STAGE,
        "check_evidence_gaps",
        "prepare_gap_research",
        "model",
        "tools",
    } <= set(subgraph.nodes)
    assert "finalize_assessment" not in subgraph.nodes


def test_gap_nodes_are_shared_by_both_assessment_stages() -> None:
    assert check_evidence_gaps.__module__ == "graph.shared"
    assert prepare_gap_research.__module__ == "graph.shared"
    assert route_model_response.__module__ == "graph.assessment_helpers"
    assert audit_subagent_module.check_evidence_gaps is check_evidence_gaps
    assert emergency_subagent_module.check_evidence_gaps is check_evidence_gaps
    assert audit_subagent_module.prepare_gap_research is prepare_gap_research
    assert (
        emergency_subagent_module.prepare_gap_research
        is prepare_gap_research
    )
    assert audit_subagent_module.route_model_response is route_model_response
    assert (
        emergency_subagent_module.route_model_response
        is route_model_response
    )


def test_run_graph_rejects_blank_question() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        run_graph("   ", chat_model=FakeChatModel([]), tools=[example_lookup])


def test_emergency_verification_uses_specific_node_update_type() -> None:
    return_type = get_type_hints(emergency_verification)["return"]

    assert return_type is EmergencyVerificationNodeUpdate
    assert get_type_hints(EmergencyVerificationNodeUpdate) == {
        "case_input": EmergencyCaseInput | None,
        "emergency_verification": EmergencyVerification | None,
        "audit_readiness": type(None),
        "assessment": EmergencyProcurementAssessment | None,
        "assessment_stage": str,
    }


def test_emergency_verification_evaluates_only_three_criteria() -> None:
    expected = verification(True)
    model = FakeChatModel([], [expected])

    update = emergency_verification(state_with_case(), chat_model=model)

    assert update["emergency_verification"] is expected
    assert isinstance(update["assessment"], EmergencyProcurementAssessment)
    assert update["assessment"].emergency_verification is expected
    assert update["assessment"].audit_readiness is None
    assert update["assessment_stage"] == EMERGENCY_VERIFICATION_STAGE
    assert len(expected.criterion_results) == 3
    assert model.structured_schemas == [EmergencyVerification]
    prompt = model.structured_inputs[0][1][1]
    assert "unexpected_event" in prompt
    assert "immediate_harm" in prompt
    assert "competition_impracticable" in prompt
    assert "approval_authority" not in prompt
    assert "2.24.090" in prompt
    assert "rules-call-001" in expected.source_ids_used


def test_audit_readiness_evaluates_only_remaining_six_and_combines_results() -> None:
    expected = audit_assessment()
    model = FakeChatModel([], [expected])
    state = state_with_case()
    verified = verification(True)
    state["emergency_verification"] = verified

    update = audit_readiness(state, chat_model=model)

    assert update["audit_readiness"] is expected
    assert len(expected.criterion_results) == 6
    assert isinstance(update["assessment"], EmergencyProcurementAssessment)
    assert update["assessment"].emergency_verification is verified
    assert update["assessment"].audit_readiness is expected
    assert len(update["assessment"].criterion_results) == 9
    assert [
        result.criterion_id for result in update["assessment"].criterion_results
    ] == [
        criterion.criterion_id
        for criterion in (*EMERGENCY_CRITERIA, *AUDIT_READINESS_CRITERIA)
    ]
    assert model.structured_schemas == [AuditReadinessAssessment]
    prompt = model.structured_inputs[0][1][1]
    assert "approval_authority" in prompt
    assert "emergency_is_verified" in prompt


def test_audit_readiness_uses_specific_node_update_type() -> None:
    return_type = get_type_hints(audit_readiness)["return"]

    assert return_type is AuditReadinessNodeUpdate
    assert get_type_hints(AuditReadinessNodeUpdate) == {
        "case_input": EmergencyCaseInput | None,
        "audit_readiness": AuditReadinessAssessment | None,
        "assessment": EmergencyProcurementAssessment | None,
        "assessment_stage": str,
    }


def test_structured_validation_error_is_returned_to_model_for_retry() -> None:
    case = load_case("EM-001")
    try:
        CriterionResult(
            criterion_id="immediate_harm",
            status=CriterionStatus.SUPPORTED,
            rationale="The harm is supported despite a material conflict.",
            conflicting_evidence=[
                EvidenceReference(
                    source_id="EM001-D01",
                    source_type="case_document",
                    description="The incident timing is materially disputed.",
                )
            ],
            confidence=0.8,
        )
    except ValidationError as error:
        validation_error = error
    else:
        pytest.fail("expected inconsistent result validation to fail")

    corrected = verification(True)
    model = FakeChatModel([], [validation_error, corrected])
    update = emergency_verification(state_with_case(), chat_model=model)

    assert update["emergency_verification"] is corrected
    assert len(model.structured_inputs) == 2
    retry_instruction = model.structured_inputs[1][-1][1]
    assert "failed Pydantic validation" in retry_instruction
    assert "immediate_harm" in retry_instruction
    assert case.case_id == corrected.case_id


def test_assessment_nodes_without_case_facts_are_safe() -> None:
    state = {"messages": [AIMessage(content="No case was requested.")]}

    verification_update = emergency_verification(state)
    audit_update = audit_readiness(
        {**state, "emergency_verification": verification(True)}
    )

    assert verification_update["emergency_verification"] is None
    assert audit_update["audit_readiness"] is None


def test_real_case_tool_artifact_reaches_emergency_verification() -> None:
    case_request = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_case_facts",
                "args": {"case_id": "EM-005"},
                "id": "case-call-005",
                "type": "tool_call",
            }
        ],
    )
    rejected = verification(False).model_copy(update={"case_id": "EM-005"})
    model = FakeChatModel(
        [case_request, AIMessage(content="Case facts retrieved.")],
        [rejected],
    )

    result = build_graph(chat_model=model, tools=[get_case_facts]).invoke(
        {
            "messages": [HumanMessage(content="Evaluate EM-005")],
            "research_rounds": 0,
            "max_research_rounds": 0,
            "gap_research_active": False,
            "gap_research_tools_used": False,
        }
    )

    tool_message = next(
        message for message in result["messages"]
        if isinstance(message, ToolMessage)
    )
    assert isinstance(tool_message.artifact, EmergencyCaseInput)
    assert tool_message.artifact.case_id == "EM-005"
    assert result["case_input"].case_id == "EM-005"
    assert result["emergency_verification"].case_id == "EM-005"
    assert result.get("audit_readiness") is None


def test_parent_graph_evaluates_dynamic_case_input_without_case_tool() -> None:
    case = EmergencyCaseInput(
        description=(
            "A critical treatment chemical may run out before the next "
            "scheduled delivery."
        )
    )
    rejected = verification(False).model_copy(update={"case_id": case.case_id})
    model = FakeChatModel([], [rejected])

    result = build_graph(
        chat_model=model,
        tools=[get_case_facts, example_lookup],
    ).invoke(
        {
            "messages": [HumanMessage(content="Evaluate the supplied case")],
            "case_input": case,
            "research_rounds": 0,
            "max_research_rounds": 0,
            "gap_research_active": False,
            "gap_research_tools_used": False,
        }
    )

    assert result["case_input"] == case
    assert result["emergency_verification"].case_id == case.case_id
    assert model.bound_model.inputs == []
    assert [
        [tool.name for tool in tool_set]
        for tool_set in model.bound_tool_sets
    ] == [
        ["get_case_facts", "example_lookup"],
        ["example_lookup"],
        ["get_case_facts", "example_lookup"],
        ["example_lookup"],
    ]
    assert not any(
        isinstance(message, ToolMessage) and message.name == "get_case_facts"
        for message in result["messages"]
    )


def test_stream_graph_yields_parent_node_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = AIMessage(content="Assessment complete.")

    class FakeGraph:
        def stream(self, state: dict[str, Any], *, stream_mode: str):
            assert state["case_input"].description == "A pump failed."
            assert stream_mode == "updates"
            yield {
                "emergency_verification_subagent": {
                    "emergency_verification": verification(False)
                }
            }
            yield {"finalize_assessment": {"messages": [final]}}

    monkeypatch.setattr("graph.workflow.build_graph", lambda **kwargs: FakeGraph())
    case = EmergencyCaseInput(description="A pump failed.")

    updates = list(stream_graph("Evaluate", case_input=case))

    assert [node for node, _ in updates] == [
        "emergency_verification_subagent",
        "finalize_assessment",
    ]
    assert updates[-1][1]["messages"] == [final]


def test_check_evidence_gaps_inspects_only_current_stage() -> None:
    verification_gap = verification(None).criterion_results[0]
    audit_gap = criterion_result(
        "approval_authority",
        missing_evidence=["Applicable approval requirements"],
    )
    state = {
        "messages": [],
        "emergency_verification": verification(None),
        "audit_readiness": audit_assessment(
            unresolved_id="approval_authority"
        ),
        "assessment_stage": EMERGENCY_VERIFICATION_STAGE,
    }

    verification_update = check_evidence_gaps(state)
    state["assessment_stage"] = AUDIT_READINESS_STAGE
    audit_update = check_evidence_gaps(state)

    assert verification_update["unresolved_criteria"] == [verification_gap]
    assert audit_update["unresolved_criteria"][0].criterion_id == audit_gap.criterion_id
    assert len(audit_update["unresolved_criteria"]) == 1


def test_parent_routes_completed_emergency_verification() -> None:
    base = {
        "assessment_stage": EMERGENCY_VERIFICATION_STAGE,
        "research_rounds": 0,
        "max_research_rounds": MAX_RESEARCH_ROUNDS,
    }

    assert route_after_emergency_verification(
        {**base, "emergency_verification": verification(True), "unresolved_criteria": []}
    ) == AUDIT_READINESS_STAGE
    assert route_after_emergency_verification(
        {**base, "emergency_verification": verification(False), "unresolved_criteria": []}
    ) == "finalize"
    assert route_after_emergency_verification(
        {**base, "emergency_verification": verification(None), "unresolved_criteria": []}
    ) == "finalize"


def test_subgraph_keeps_indeterminate_verification_inside_gap_loop() -> None:
    unresolved = verification(None)
    state = {
        "assessment_stage": EMERGENCY_VERIFICATION_STAGE,
        "emergency_verification": unresolved,
        "unresolved_criteria": unresolved.criterion_results[:1],
        "research_rounds": 0,
        "max_research_rounds": MAX_RESEARCH_ROUNDS,
    }

    assert route_emergency_verification_gaps(state) == "research"
    assert route_emergency_verification_gaps(
        {
            **state,
            "research_rounds": MAX_RESEARCH_ROUNDS,
        }
    ) == "complete"
    assert route_emergency_verification_gaps(
        {**state, "emergency_verification": verification(False)}
    ) == "complete"


def test_audit_subgraph_router_handles_bounded_research() -> None:
    assert route_audit_readiness_gaps(
        {
            "unresolved_criteria": [
                criterion_result("approval_authority")
            ],
            "research_rounds": 0,
            "max_research_rounds": MAX_RESEARCH_ROUNDS,
        }
    ) == "research"


def test_prepare_gap_research_batches_all_current_stage_gaps() -> None:
    unresolved = [
        criterion_result(
            "approval_authority",
            missing_evidence=["Approval evidence"],
        ),
        criterion_result(
            "price_reasonableness",
            missing_evidence=["Vendor quote"],
        ),
    ]
    state = {
        "messages": [],
        "assessment_stage": AUDIT_READINESS_STAGE,
        "unresolved_criteria": unresolved,
        "research_rounds": 0,
        "max_research_rounds": MAX_RESEARCH_ROUNDS,
    }

    update = prepare_gap_research(state)
    context = str(update["messages"][0].content)

    assert update["research_rounds"] == 1
    assert context.count('"criterion_id"') == 2
    assert "audit_readiness" in context
    assert "approval_authority" in context
    assert "price_reasonableness" in context


def test_html_list_formatter_closes_lists_and_escapes_values() -> None:
    lines = ["Before"]

    append_html_list(
        lines,
        "Missing evidence",
        ["Vendor <quote>", "Approval & date"],
    )

    assert lines == [
        "Before",
        "",
        "<h4>Missing evidence</h4>",
        "<ul>",
        "<li>Vendor &lt;quote&gt;</li>",
        "<li>Approval &amp; date</li>",
        "</ul>",
        "",
    ]


def test_finalizer_renders_all_verification_results_and_case_context() -> None:
    rejected = rejected_em003_verification()
    rejected.source_ids_used.append("call_uiyExxI0n50LIITkllkxv2na")
    state = state_with_case("EM-003")
    state.update(
        {
            "assessment_stage": EMERGENCY_VERIFICATION_STAGE,
            "emergency_verification": rejected,
            "audit_readiness": None,
        }
    )

    update = finalize_assessment(state)
    content = str(update["messages"][0].content)

    assert "EM-003 — Contract Expiration Caused by Poor Planning" in content
    assert "emergency procurement exception is not justified" in content
    assert rejected.rationale in content
    assert "unexpected_event" in content
    assert "Status: not_supported" in content
    assert "immediate_harm" in content
    assert "Status: supported" in content
    assert "competition_impracticable" in content
    assert "Status: partially_supported" in content
    assert "<h4>Missing evidence</h4>" in content
    assert "<h4>Follow-up questions</h4>" in content
    assert "<ul>" in content
    assert "</ul>" in content
    assert "Documented review of available alternatives" in content
    assert "Were alternate providers contacted?" in content
    assert "EM003-D02" in content
    assert "Sources used" not in content
    assert "call_uiyExxI0n50LIITkllkxv2na" not in content


def test_finalizer_renders_audit_result_and_existing_checklist() -> None:
    audit = audit_assessment(unresolved_id="post_facto_formalization")
    audit.missing_documents = ["Signed emergency justification"]
    audit.required_approvals = ["City Manager approval"]
    audit.next_steps = ["Add the executed approval to the procurement file"]
    audit.requires_human_review = True
    audit.human_review_reason = "The approval must be confirmed by the agency."
    audit.audit_risks = [
        AuditRisk(
            risk_id="missing_approval",
            title="Approval not documented",
            description="The current file does not contain the approval.",
            severity="high",
            related_criterion_ids=["approval_authority"],
            recommended_action="Obtain and retain the executed approval.",
        )
    ]
    state = state_with_case("EM-001")
    state.update(
        {
            "assessment_stage": AUDIT_READINESS_STAGE,
            "emergency_verification": verification(True),
            "audit_readiness": audit,
        }
    )

    update = finalize_assessment(state)
    content = str(update["messages"][0].content)

    assert "## Audit readiness" in content
    assert "not yet audit-ready" in content
    assert "Recommendation: additional_evidence_required" in content
    assert "post_facto_formalization" in content
    assert "Status: insufficient evidence — criterion not passed" in content
    assert "Status: not_evaluated" not in content
    assert "Signed emergency justification" in content
    assert "City Manager approval" in content
    assert "Add the executed approval" in content
    assert "Approval not documented" in content
    assert audit.human_review_reason in content


def test_finalizer_uses_explicit_stage_when_both_results_exist() -> None:
    state = state_with_case()
    state.update(
        {
            "assessment_stage": AUDIT_READINESS_STAGE,
            "emergency_verification": verification(False),
            "audit_readiness": audit_assessment(),
        }
    )

    content = str(finalize_assessment(state)["messages"][0].content)

    assert "## Audit readiness" in content
    assert "## Emergency verification" not in content


def test_verified_emergency_runs_audit_readiness(monkeypatch) -> None:
    verification_calls = 0
    audit_calls = 0

    def fake_verification(state: dict[str, Any], **_: Any) -> dict[str, Any]:
        nonlocal verification_calls
        verification_calls += 1
        return {
            "emergency_verification": verification(True),
            "assessment_stage": EMERGENCY_VERIFICATION_STAGE,
        }

    resolved_audit = audit_assessment()

    def fake_audit(state: dict[str, Any], **_: Any) -> dict[str, Any]:
        nonlocal audit_calls
        audit_calls += 1
        return {
            "audit_readiness": resolved_audit,
            "assessment": EmergencyProcurementAssessment(
                case_id="EM-001",
                emergency_verification=state["emergency_verification"],
                audit_readiness=resolved_audit,
            ),
            "assessment_stage": AUDIT_READINESS_STAGE,
        }

    monkeypatch.setattr(
        emergency_subagent_module,
        "emergency_verification",
        fake_verification,
    )
    monkeypatch.setattr(audit_subagent_module, "audit_readiness", fake_audit)
    model = FakeChatModel([AIMessage(content="Initial research complete.")])

    result = build_graph(chat_model=model, tools=[example_lookup]).invoke(
        {"messages": [HumanMessage(content="Evaluate the case")]}
    )

    assert verification_calls == 1
    assert audit_calls == 1
    assert result["audit_readiness"] is resolved_audit


def test_rejected_emergency_finalizes_without_audit(monkeypatch) -> None:
    audit_calls = 0

    def fake_verification(state: dict[str, Any], **_: Any) -> dict[str, Any]:
        return {
            "emergency_verification": verification(False),
            "assessment_stage": EMERGENCY_VERIFICATION_STAGE,
        }

    def fake_audit(state: dict[str, Any], **_: Any) -> dict[str, Any]:
        nonlocal audit_calls
        audit_calls += 1
        return {}

    monkeypatch.setattr(
        emergency_subagent_module,
        "emergency_verification",
        fake_verification,
    )
    monkeypatch.setattr(audit_subagent_module, "audit_readiness", fake_audit)
    model = FakeChatModel([AIMessage(content="Initial research complete.")])

    result = build_graph(chat_model=model, tools=[example_lookup]).invoke(
        {"messages": [HumanMessage(content="Evaluate the case")]}
    )

    assert audit_calls == 0
    assert result["emergency_verification"].emergency_is_verified is False
    final_content = str(result["messages"][-1].content)
    assert "emergency procurement exception is not justified" in final_content
    assert "unexpected_event" in final_content
    assert "immediate_harm" in final_content
    assert "competition_impracticable" in final_content


def test_em003_poor_planning_runs_inside_subgraph_and_skips_audit() -> None:
    rejected = rejected_em003_verification()
    case_request = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_case_facts",
                "args": {"case_id": "EM-003"},
                "id": "em003-case-facts",
                "type": "tool_call",
            }
        ],
    )
    model = FakeChatModel(
        [case_request, AIMessage(content="Case research complete.")],
        [rejected],
    )

    result = build_graph(chat_model=model, tools=[get_case_facts]).invoke(
        {"messages": [HumanMessage(content="Evaluate case EM-003")]}
    )

    assert result["emergency_verification"] == rejected
    assert result.get("audit_readiness") is None
    assert model.structured_schemas == [EmergencyVerification]
    assert [
        message.tool_call_id
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ] == ["em003-case-facts"]
    final_content = str(result["messages"][-1].content)
    assert "emergency procurement exception is not justified" in final_content
    assert "missed procurement planning" in final_content


def test_verification_gap_research_returns_to_verification(monkeypatch) -> None:
    verifications = iter([verification(None), verification(True)])
    verification_calls = 0

    def fake_verification(state: dict[str, Any], **_: Any) -> dict[str, Any]:
        nonlocal verification_calls
        verification_calls += 1
        return {
            "emergency_verification": next(verifications),
            "assessment_stage": EMERGENCY_VERIFICATION_STAGE,
        }

    resolved_audit = audit_assessment()

    def fake_audit(state: dict[str, Any], **_: Any) -> dict[str, Any]:
        return {
            "audit_readiness": resolved_audit,
            "assessment": EmergencyProcurementAssessment(
                case_id="EM-001",
                emergency_verification=state["emergency_verification"],
                audit_readiness=resolved_audit,
            ),
            "assessment_stage": AUDIT_READINESS_STAGE,
        }

    monkeypatch.setattr(
        emergency_subagent_module,
        "emergency_verification",
        fake_verification,
    )
    monkeypatch.setattr(audit_subagent_module, "audit_readiness", fake_audit)
    model = FakeChatModel(
        [
            AIMessage(content="Initial research complete."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "example_lookup",
                        "args": {"query": "foreseeability evidence"},
                        "id": "verification-gap-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "example_lookup",
                        "args": {"query": "competition evidence"},
                        "id": "verification-gap-2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="Gap research complete."),
        ]
    )

    result = build_graph(chat_model=model, tools=[example_lookup]).invoke(
        {"messages": [HumanMessage(content="Evaluate the case")]}
    )

    assert verification_calls == 2
    assert result["research_rounds"] == 1
    assert result["emergency_verification"].emergency_is_verified is True
    assert result["audit_readiness"] is resolved_audit
    assert [
        message.tool_call_id
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ] == ["verification-gap-1", "verification-gap-2"]


def test_audit_gap_research_returns_to_audit(monkeypatch) -> None:
    initial_audit = audit_assessment(unresolved_id="approval_authority")
    resolved_audit = audit_assessment()
    audits = iter([initial_audit, resolved_audit])
    audit_calls = 0

    def fake_verification(state: dict[str, Any], **_: Any) -> dict[str, Any]:
        return {
            "emergency_verification": verification(True),
            "assessment_stage": EMERGENCY_VERIFICATION_STAGE,
        }

    def fake_audit(state: dict[str, Any], **_: Any) -> dict[str, Any]:
        nonlocal audit_calls
        audit_calls += 1
        current = next(audits)
        return {
            "audit_readiness": current,
            "assessment": EmergencyProcurementAssessment(
                case_id="EM-001",
                emergency_verification=state["emergency_verification"],
                audit_readiness=current,
            ),
            "assessment_stage": AUDIT_READINESS_STAGE,
        }

    monkeypatch.setattr(
        emergency_subagent_module,
        "emergency_verification",
        fake_verification,
    )
    monkeypatch.setattr(audit_subagent_module, "audit_readiness", fake_audit)
    model = FakeChatModel(
        [
            AIMessage(content="Initial research complete."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "example_lookup",
                        "args": {"query": "local authority"},
                        "id": "gap-call-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "example_lookup",
                        "args": {"query": "state authority"},
                        "id": "gap-call-2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="Gap research complete."),
        ]
    )

    result = build_graph(chat_model=model, tools=[example_lookup]).invoke(
        {"messages": [HumanMessage(content="Evaluate the case")]}
    )

    assert audit_calls == 2
    assert result["audit_readiness"] is resolved_audit
    assert [
        message.tool_call_id
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ] == ["gap-call-1", "gap-call-2"]


def test_audit_subgraph_preserves_unresolvable_agency_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unresolved_audit = audit_assessment(
        unresolved_id="post_facto_formalization"
    )

    def fake_audit(state: dict[str, Any], **_: Any) -> dict[str, Any]:
        return {
            "audit_readiness": unresolved_audit,
            "assessment": EmergencyProcurementAssessment(
                case_id="EM-001",
                emergency_verification=state["emergency_verification"],
                audit_readiness=unresolved_audit,
            ),
            "assessment_stage": AUDIT_READINESS_STAGE,
        }

    monkeypatch.setattr(audit_subagent_module, "audit_readiness", fake_audit)
    no_tool_response = AIMessage(
        content="The signed file must be supplied by the agency."
    )
    model = FakeChatModel([no_tool_response])
    initial_state = state_with_case()
    initial_state.update(
        {
            "emergency_verification": verification(True),
            "audit_readiness": None,
            "assessment": None,
            "assessment_stage": AUDIT_READINESS_STAGE,
            "unresolved_criteria": [],
        }
    )

    result = build_audit_readiness_subgraph(
        chat_model=model,
        tools=[example_lookup],
    ).invoke(initial_state)

    assert result["audit_readiness"] is unresolved_audit
    assert result["research_rounds"] == 1
    assert [
        item.criterion_id for item in result["unresolved_criteria"]
    ] == ["post_facto_formalization"]
    assert result["messages"][-1] is no_tool_response
    assert not any(
        isinstance(message, ToolMessage)
        and message.name == "example_lookup"
        for message in result["messages"]
    )


def test_audit_subgraph_research_stops_after_three_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unresolved_audit = audit_assessment(unresolved_id="approval_authority")
    audit_calls = 0

    def fake_audit(state: dict[str, Any], **_: Any) -> dict[str, Any]:
        nonlocal audit_calls
        audit_calls += 1
        return {
            "audit_readiness": unresolved_audit,
            "assessment": EmergencyProcurementAssessment(
                case_id="EM-001",
                emergency_verification=state["emergency_verification"],
                audit_readiness=unresolved_audit,
            ),
            "assessment_stage": AUDIT_READINESS_STAGE,
        }

    monkeypatch.setattr(audit_subagent_module, "audit_readiness", fake_audit)
    responses: list[AIMessage] = []
    for round_number in range(1, MAX_RESEARCH_ROUNDS + 1):
        responses.extend(
            [
                tool_request(
                    call_id=f"audit-round-{round_number}",
                    query=f"audit round {round_number}",
                ),
                AIMessage(
                    content=f"Audit research round {round_number} complete."
                ),
            ]
        )
    model = FakeChatModel(responses)
    initial_state = state_with_case()
    initial_state.update(
        {
            "emergency_verification": verification(True),
            "audit_readiness": None,
            "assessment": None,
            "assessment_stage": AUDIT_READINESS_STAGE,
            "unresolved_criteria": [],
        }
    )

    result = build_audit_readiness_subgraph(
        chat_model=model,
        tools=[example_lookup],
    ).invoke(initial_state)

    assert result["research_rounds"] == MAX_RESEARCH_ROUNDS == 3
    assert audit_calls == 4
    assert [
        item.criterion_id for item in result["unresolved_criteria"]
    ] == ["approval_authority"]
    assert sum(
        isinstance(message, ToolMessage)
        and message.name == "example_lookup"
        for message in result["messages"]
    ) == 3


def test_model_can_decline_gap_research_and_preserve_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unresolved_verification = verification(None)

    def fake_verification(state: dict[str, Any], **_: Any) -> dict[str, Any]:
        return {
            "emergency_verification": unresolved_verification,
            "assessment_stage": EMERGENCY_VERIFICATION_STAGE,
        }

    monkeypatch.setattr(
        emergency_subagent_module,
        "emergency_verification",
        fake_verification,
    )
    no_tool_response = AIMessage(
        content="The remaining evidence must be provided by the agency."
    )
    model = FakeChatModel(
        [AIMessage(content="Initial research complete."), no_tool_response]
    )

    result = build_graph(chat_model=model, tools=[example_lookup]).invoke(
        {"messages": [HumanMessage(content="Evaluate the case")]}
    )

    assert result["messages"][-2] is no_tool_response
    final_content = str(result["messages"][-1].content)
    assert "insufficient to determine" in final_content
    assert "Additional emergency evidence" in final_content
    assert len(result["unresolved_criteria"]) == 1
    assert result["research_rounds"] == 1
    assert not any(
        isinstance(message, ToolMessage) for message in result["messages"]
    )


def test_gap_research_stops_after_three_rounds(monkeypatch) -> None:
    unresolved_verification = verification(None)
    verification_calls = 0

    def fake_verification(state: dict[str, Any], **_: Any) -> dict[str, Any]:
        nonlocal verification_calls
        verification_calls += 1
        return {
            "emergency_verification": unresolved_verification,
            "assessment_stage": EMERGENCY_VERIFICATION_STAGE,
        }

    monkeypatch.setattr(
        emergency_subagent_module,
        "emergency_verification",
        fake_verification,
    )
    responses: list[AIMessage] = [AIMessage(content="Initial research complete.")]
    for round_number in range(1, MAX_RESEARCH_ROUNDS + 1):
        responses.extend(
            [
                tool_request(
                    call_id=f"round-{round_number}",
                    query=f"round {round_number}",
                ),
                AIMessage(content=f"Research round {round_number} complete."),
            ]
        )
    model = FakeChatModel(responses)

    result = build_graph(chat_model=model, tools=[example_lookup]).invoke(
        {"messages": [HumanMessage(content="Evaluate the case")]}
    )

    assert result["research_rounds"] == MAX_RESEARCH_ROUNDS == 3
    assert verification_calls == 4
    assert len(result["unresolved_criteria"]) == 1
    assert sum(
        isinstance(message, ToolMessage) for message in result["messages"]
    ) == 3
